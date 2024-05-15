# -*- coding: utf-8 -*-
"""
A tree structure that inherits from nlkt's ParentedTree that makes
it possible to index leaves so that for every subtree the span can
be identified.
"""
from nltk.tree import ParentedTree

from .errors.errors import UnindexedTree


class IndexedTree(ParentedTree):

    def __init__(self, label, children=None):
        super().__init__(label, children)
        self.indexed = False

    def index(self):
        """Transforms every leaf into tuple of token and its index.
        (NP (DET the) (N dog)) -> (NP (DET the/0) (N dog/1))
        """
        if not self.indexed:
            # Indices for leaves.
            positions = self.treepositions("leaves")
            for idx, positions in enumerate(positions):
                # str(idx) because repr/str methods won't work otherwise.
                # (the problem lies somewhere in nltk's string methods)
                self[positions] = (self[positions], str(idx))
            self._set_indexed()

    def _set_indexed(self):
        """Sets attribute indexed recursively"""
        self.indexed = True
        for child in self:
            if isinstance(child, IndexedTree):
                child._set_indexed()

    def _token(self, leaf):
        """Returns the word of a leaf (str)"""
        return leaf[0]

    def _index(self, leaf):
        """Returns the index of a leaf (int)"""
        return int(leaf[1])

    def span(self):
        """Returns the span of an IndexedTree.
        The span is a tuple of the index of the first leaf
        and the index of the last leaf + 1

        Returns: 2-tuple of integers

        Raises: Exception if tree is not indexed yet.
        """
        if self.indexed:
            leaves = self.leaves(True)
            start = self._index(leaves[0])
            end = self._index(leaves[-1]) + 1
            return start, end
        raise UnindexedTree("Tree is not indexed yet. Call index().")

    def leaves(self, indexed=False):
        """Returns the leaves of a tree. Optionally, without the indices."""
        leaves = []
        for child in self:
            if isinstance(child, IndexedTree):
                leaves.extend(child.leaves(indexed))
            else:
                if indexed:
                    leaves.append(child)
                else:
                    leaves.append(self._token(child))
        return leaves

    def pos(self):
        pos = super().pos()
        if not self.indexed:
            return pos
        return [(self._token(tok), p) for tok, p in pos]
