# -*- coding: utf-8 -*-
"""
A class that represents clusters in a document.
Clusters are essentially a disjoint-set structure.
"""
from collections import deque
import logging

from .indexed_tree import IndexedTree
from .mention import Mention


class Clusters:

    RE = {"NP", "PRP$"}

    def __init__(self, doc):
        """Constructor of a Clusters instance.

        Args:
            doc (OntonotesDoc): A OntonotesDoc object.
        """
        self._doc = doc
        self.clusters = dict()
        # Maps mention ids to Mention objects.
        self.mentions = dict()

        self._intialize()

    def _intialize(self):
        """Intializes the clusters by extracting mentions and
        assigning each a cluster."""
        for sentence in self._doc:
            tree = sentence.tree()
            # Subtrees are filtered for referential expressions.
            for subt in tree.subtrees(filter=lambda t: t.label() in self.RE):
                start, end = subt.span()
                mention_id = (sentence.index, start, end)
                mention = Mention(mention_id, subt)
                self.clusters[mention] = {mention}
                self.mentions[mention_id] = mention

    def sentence(self, mention):
        """Returns the Sentence object in which the mention was found.

        Args:
            mention (Mention): A mention object.
        """
        return self._doc[mention.index()]

    def find(self, mention):
        """Finds the current representative mention of the cluster."""
        return mention.pointer

    def merge(self, first, second):
        """Performs union of two clusters of mentions
        if they belong two different clusters.

        After a merge, the representative of one mention
        will not be present in self.clusters. Its value will have been merged
        with the value of the other mention's representative

        Args:
            first (Mention): A Mention object.
            second (Mention): A Mention object.

        Returns:
            True if the clusters of the mentions have been merged.
            If the mentions were already in the same cluster, returns False.
        """
        repr_first = self.find(first)
        repr_second = self.find(second)
        # Both mentions are already in the same cluster.
        if repr_first == repr_second:
            return False
        # This means it appear earlier as the id specifies its position.
        if repr_first.id < repr_second.id:
            earlier = repr_first
            latter = repr_second
        else:
            earlier = repr_second
            latter = repr_first
        # Perfom union of mentions and delete second (later) cluster.
        second_mentions = self.clusters.pop(latter)
        first_mentions = self.clusters[earlier]
        self.clusters[earlier] = first_mentions.union(second_mentions)
        # Set new pointers for all mentions in second cluster
        for mention in second_mentions:
            mention.pointer = earlier
        return True

    def unresolved(self):
        """Returns a sorted list of mentions that are the first mention
        in their cluster. These are the mentions that are its clusters
        representative."""
        # Mentions are sorted by sentence index first,
        # start and end index second.
        return sorted(self.clusters.keys(), key=lambda m: m.id)

    def antecedents(self, mention):
        """Returns a list of mentions that represent the mention's
        antecedents.

        Args:
            mention (Mention): A Mention object to extract antecedents from.

        Returns: List of Mention objects.
        """
        # We have seen this mention before and have extracted its antecedents.
        if mention.antecedents:
            return mention.antecedents
        # We have not seen it before and therefore generate its antecedents.
        antecedents = self._generate_antecedents(mention)
        # Set antecedents for mention to avoid unnecessary computations.
        mention.antecedents = antecedents
        return antecedents

    def _generate_antecedents(self, mention):
        """Generates antecedents for a mention according to its
        properties.

        Antecedents are extracted from the sentence the mentions appears in
        and the previous sentence.
        Antecedents are sorted by breadth-first search. Left-to-right in the
        same sentence and right-to-left in the previous sentence if the mention
        is not pronominal.
        """
        index, start, end = mention.id
        prev = index - 1
        # Get mentions in previous sentence
        prev_sent = []
        if prev >= 0:
            prev_tree = self._doc[prev].tree()
            # If a mention is pronominal, we search left to right,
            # otherwise we search right to left.
            prev_sent = self._bfs(prev_tree,
                                  prev,
                                  left_to_right=mention.pronominal)
        # Get mentions in same sentence
        tree = self._doc[index].tree()
        same_sent = self._bfs(tree, index, mention=mention)
        # First antecedents of same, then of previous sentence.
        return list(same_sent) + list(prev_sent)

    def _bfs(self, tree, index, left_to_right=True, mention=None):
        """A generator that yields Mention objects by searching
        a tree structure breadth-first.

        Args:
            tree (IndexedTree):
                A IndexedTree that should be searched.
            index (int):
                An integer that represents the index of the sentence
                the tree structure is from.
            left_to_right (bool):
                If True, children will be searched from left to right.
                If False, children will be searched from right to left.
            mention (Mention):
                If given, search will be stopped when mention is reached.
        """
        queue = deque()
        queue.append(tree)
        while queue:
            next_tree = queue.popleft()
            if next_tree.label() in self.RE:
                span = next_tree.span()
                if mention:
                    # If this is the case we have reached the mention,
                    # and don't need to look for more antecedents.
                    if span == mention.span():
                        return
                try:
                    # Map found id to Mention object.
                    yield self.mentions[(index, *span)]
                except KeyError:
                    logging.warning(f"Unseen mention {(index, *span)} "
                                    "in {self._doc.path} found.")
                    continue
            if not left_to_right:
                next_tree = reversed(next_tree)
            for child in next_tree:
                if isinstance(child, IndexedTree):
                    queue.append(child)

    def __str__(self):
        string = f"<ClustersObject with {len(self.clusters)} Clusters>\n"
        return string

    def __getitem__(self, mention):
        """Returns the cluster (set) a mention is currently assigned to."""
        if isinstance(mention, tuple):
            mention = self.mentions[mention]
        repr_ment = self.find(mention)
        return self.clusters[repr_ment]

    def __iter__(self):
        return iter(self.clusters)
