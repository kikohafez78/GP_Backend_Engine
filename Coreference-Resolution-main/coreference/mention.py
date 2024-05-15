# -*- coding: utf-8 -*-
"""
Class that represents a mention aka a referential expression.
A mention is definded by a unique id consisting of (sentence_index, start, end)
"""
from .errors.errors import UnindexedTree
from .indexed_tree import IndexedTree


class Mention:

    PRONOUNS = {"PRP", "DT", "PRP$"}
    NOMINAL = {"NN", "NP", "NNS", "NNP", "NNPS"}
    INDEFINITE = {"a",
                  "an",
                  "some",
                  "no",
                  "most",
                  "any",
                  "few",
                  "many",
                  "several",
                  "there"}

    def __init__(self, mention_id, tree):
        """Constructor for a Mention instance.

        Args:
            mention_id (tuple):
                A three tuple where the first element is the sentence index,
                the second is the start and the third is the end of a mention.
            tree (IndexedTree):
                A Indexed Tree object that specifies the syntactic structure
                of the mention.
        """
        self.id = mention_id
        # Points to the representative mention of the current cluster.
        # Initially, this is the mention itself.
        self.pointer = self
        # Check if tree is compatible.
        if not isinstance(tree, IndexedTree):
            raise TypeError("Expected an IndexedTree.")
        if not tree.indexed:
            raise UnindexedTree("IndexedTree is not indexed.")
        self.tree = tree
        self.antecedents = None
        self._pronominal = self._is_pronominal(self.tree)
        self._indefinite = self._is_indefinite(self.tree)
        self._head = self._get_head(self.tree)
        self._words = self.tree.leaves()
        self._pos = self.tree.pos()

    @property
    def words(self):
        """Returns the token of a mention in a list."""
        return self._words

    @property
    def pos(self):
        """Returns the pos tags of a mention in a list."""
        return self._pos

    @property
    def pronominal(self):
        """Returns True if mention is a pronoun, False otherwise."""
        return self._pronominal

    @property
    def indefinite(self):
        """Returns True if mention is considered an indefinite NP.
        False otherwise."""
        return self._indefinite

    @property
    def head(self):
        """Returns a token (str) that is considered the head noun of the mention."""
        return self._head

    def span(self):
        "Returns a tuple consisting of the start and end index of the mention."
        return self.id[1], self.id[2]

    def index(self):
        "Returns the index of the sentence that contains the mention."
        return self.id[0]

    def _is_pronominal(self, tree):
        """Determines whether a mention is a pronoun or not.

        A mention is considered a pronoun if it has a unary tree structure
        and there is a pronoun label (PRP) in its tree structure.

        Args:
            tree (IndexedTree): An IndexedTree object.

        Returns: Boolean
        """
        # A pronoun is contained in a tree where each
        # node has exactly one child.
        if len(tree) == 1:
            child = tree[0]
            # Accounts for cases like: (PRP$ his)
            if tree.label() in self.PRONOUNS:
                return True
            # Accounts for cases like: (NP (PRP he))
            elif isinstance(child, IndexedTree):
                # This is recursive because pronouns can have structure:
                # (NP (NP (PRP it)))
                return self._is_pronominal(child)
        return False

    def _is_indefinite(self, tree):
        """Determines whether a mention is indefinite or not.

        A mention is considered indefinite if its first word
        is an indefinite determiner.

        Args:
            tree (IndexedTree): An IndexedTree object.

        Returns: Boolean
        """
        # Base case: Reached a leaf.
        if not isinstance(tree, IndexedTree):
            # Leaf token is an indefinite determiner.
            if tree[0].lower() in self.INDEFINITE:
                return True
            # Obviously, this is a very relaxed heuristic,
            # e.g. bare NPs will always be interpreted as definite.
            else:
                return False
        # If there is a determiner, it is most likely in the first
        # constituent.
        return self._is_indefinite(tree[0])

    def _get_head(self, tree):
        """Extracts the head noun of a mention.

        The head is contained in the right most nominal child of
        a tree structure. This is done recursively until a leaf is reached.

        Args:
            tree (IndexedTree): An IndexedTree object.

        Returns: str
        """
        # Base case: tree is a leaf.
        if not isinstance(tree, IndexedTree):
            # Leaves are tuples of (token, index)
            return tree[0]
        # Because of right headedness, we search for the first
        # nominal constituent from right to left in all children.
        for i in range(len(tree)-1, -1, -1):
            child = tree[i]
            if isinstance(child, IndexedTree) and child.label() in self.NOMINAL:
                return self._get_head(child)
        # This is a fall back: As english has right headedness in nouns,
        # the naive approach is to assume the last child contains the head.
        return self._get_head(tree[-1])

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        return self.id == other

    def __repr__(self):
        return f"Mention{self.id}"

    def __str__(self):
        words = " ".join(self.words)
        return f"<Mention{self.id}, '{words}'>"
