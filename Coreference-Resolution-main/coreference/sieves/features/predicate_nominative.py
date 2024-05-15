# -*- coding: utf-8 -*-
"""
A feature that determines if an antecedent and a mention
are in an predicate nominative construction
"""

from ..abstract_sieve import AbstractClusterFeature
from ...indexed_tree import IndexedTree


class PredicateNominative(AbstractClusterFeature):

    SENT = {"S", "SBAR"}
    VP = {"VP"}

    def __init__(self, predicate=("be",)):
        """Constructor of PredicateNominative instance.

        Args:
            predicate:
                Iterable of strings that are considered predicates.
        """
        self.predicate = predicate

    def __call__(self, cluster, antecedent, mention):
        """Determines if a mention and its antecedent are in
        an predicate nominative construction.

        A predicate nominative construction is assumed if the
        antecedent is the subject of a sentence (directly headed by an S node)
        and the mention is the object (directly headed by a VP node), there is
        only one verb in the VP and it is a valid predicate
        (specified by self.predicate).

        Args:
            cluster (Clusters):
                A Clusters object that should contain the given mentions.
            antecedent (Mention):
                A Mention object that represents a mention that appears
                before the given mention.
            mention (Mention): Any Mention object.

        Returns:
            True if all of the above applies, False otherwise.
        """
        # Get the sentence object that contains the mention
        sentence = cluster.sentence(mention)
        if antecedent.index() == mention.index():
            # The mention should be in the VP (object)
            ment_parent = self._get_parent(mention, self.VP)
            # The antecedent should be the subject, so its parent
            # should be an S/SBAR node
            ant_parent = self._get_parent(antecedent, self.SENT)
            if ment_parent is not None and ant_parent is not None:
                # Making sure the VP that contains the mention,
                # is the VP of the S/SBAR tree of the antecedent.
                if ment_parent.parent() is ant_parent:
                    verb = self._get_verb(ment_parent)
                    if verb:
                        # Leaves are tuples of token and index.
                        _, index = verb
                        # Check if extracted verb is a predicate.
                        lemma = sentence.lemma()
                        if lemma[int(index)] in self.predicate:
                            return True
        return False

    def _get_verb(self, vp):
        """Determines the only verb of a tree structure.

        Args:
            vp (IndexedTree):
                A tree structure that represents a verb phrase.

        Returns:
            None if there is more than one verb in the vp.
            A leaf of an IndexedTree (tuple of token, index) that
            represents the only verb in the vp otherwise.
        """
        # Base case: reached a leaf.
        if not isinstance(vp, IndexedTree):
            return vp
        n_verbs = 0
        index_verb = 0
        for idx, child in enumerate(vp):
            # There should be only one verb in the VP.
            if n_verbs > 1:
                return None
            if isinstance(child, IndexedTree) and child.label().startswith("V"):
                index_verb = idx
                n_verbs += 1
        return self._get_verb(vp[index_verb])

    def _get_parent(self, mention, parent_label):
        """Determines if the mention's parent has a specified label.
        Returns the parent if that is the case. None otherwise.

        Args:
            mention (Mention): Any Mention object.
            parent_label:
                Iterable of strings representing valid
                labels for the parent.
        """
        parent = mention.tree.parent()
        if parent is None or parent.label() not in parent_label:
            return None
        return parent
