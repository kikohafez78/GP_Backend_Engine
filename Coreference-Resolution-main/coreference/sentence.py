# -*- coding: utf-8 -*-
"""
A data structure that represents a sentence with informations
like syntactic structure, lemma and coreference informations.
"""
import logging

from .indexed_tree import IndexedTree


class Sentence:

    DOC_COL = 0
    WORD_ID_COL = 2
    WORD_COL = 3
    POS_COL = 4
    TREE_COL = 5
    LEMMA_COL = 6
    NE_COL = 10
    COREF_COL = -1

    def __init__(self, sent_id, conll_sentence):
        self._id = sent_id
        self._words = []
        self._tree = None
        self._pos = []
        self._lemma = []
        self._ne = dict()
        self._coref = dict()

        self._process(conll_sentence)

    @property
    def index(self):
        """Returns the index of sentence."""
        return self._id

    def named_entities(self):
        """Returns a dictionary that contains named entity information.

        Returns:
            Dict. Keys are the start and end indices of a named
            entity. The values represent the type of named entity, e.g. PERSON.
        """
        return self.ne

    def coreference(self):
        """Returns a dictionary that contains coreference informations.

        Returns:
            Dict. Keys are abitrary ids and value is a list of tuples where
            each tuple consists of a start and end index for that mention.
        """
        return self._coref

    def tree(self):
        """Returns IndexedTree of a sentence."""
        return self._tree

    def words(self, tagged=True):
        """Returns tokens of a sentence in a list.
        If tagged is True, list contains tuples of tokens and pos tags."""
        if tagged:
            return list(zip(self._words, self._pos))
        return self._words

    def lemma(self, token=False):
        """Returns list of lemmas for each word in sentence.
        If there was no lemma specified, lemma is None.

        Args:
            token (bool):
                If True, returned list will contain tuple of
                token and lemma. Otherwise the list will contain
                only the token.
        """
        if token:
            return list(zip(self._words, self._lemma))
        return self._lemma

    def _process(self, conll_sentence):
        """Read in a sentence in the CoNLL format and
        set instance attributes.

        Args:
            conll_sentence (list):
                A list of lists where each list represents
                a conll line with columns (str).

        Returns: None
        """
        tree_str = ""
        coref_stack = dict()
        ne_stack = []
        for line in conll_sentence:
            doc = line[self.DOC_COL]
            word = line[self.WORD_COL]
            pos = line[self.POS_COL]
            word_id = int(line[self.WORD_ID_COL])
            coref = line[self.COREF_COL]
            ne = line[self.NE_COL]
            lemma = line[self.LEMMA_COL]
            if lemma == "-":
                lemma = None
            self._lemma.append(lemma)
            self._words.append(word)
            self._pos.append(pos)
            word_pos = f"({pos} {word})"
            tree_str += line[self.TREE_COL].replace("*", word_pos)
            # Read coreference information.
            self._update_coref(coref, word_id, coref_stack)
            # Read named entity information.
            self._update_ne(ne, word_id, ne_stack)
        # Some tree strings from ontonotes seem to be malformed
        try:
            self._tree = IndexedTree.fromstring(tree_str)
        except ValueError:
            logging.debug(f"Malformed tree structure in {doc} "
                          f"in sentence {self._id}")
            # Replace by a rudimentary tree structure.
            self._tree = IndexedTree("NONE", self._words)
        finally:
            # Indexing the tree helps to identify spanning subtrees.
            self._tree.index()

    def _update_coref(self, coref, word_id, stack):
        """Update the current coreference stack with
        new info from the coreference column."""
        if not coref.startswith("-"):
            # Coreference infos that start/end in the same line
            # are seperated by |. E.g.(2|(89
            coref_split = coref.split("|")
            for coref in coref_split:
                # The id that specifies which mentions are coreferential.
                coref_id = int(coref.strip("(").strip(")"))
                # A coreference info starts.
                if coref.startswith("("):
                    stack.setdefault(coref_id, [])
                    stack[coref_id].append(word_id)
                # A coreference info that ends.
                if coref.endswith(")"):
                    start = stack[coref_id].pop()
                    self._coref.setdefault(coref_id, [])
                    self._coref[coref_id].append((start, word_id+1))

    def _update_ne(self, ne, word_id, stack):
        """Update the current named entity stack with
        new info from the named entity column."""
        if ne.startswith("*"):
            # A named entity ends.
            if ne.endswith(")"):
                ne_type, start = stack.pop()
                self._ne[(start, word_id+1)] = ne_type
        else:
            ne = ne.strip("(").strip(")")
            # A named entity starts and ends in same line.
            if not ne.endswith("*"):
                self._ne[(word_id, word_id+1)] = ne
            else:
                # A named entity starts.
                ne = ne.strip("*")
                stack.append((ne, word_id))

    def __getitem__(self, index):
        return self._words[index]

    def __str__(self):
        words = str(self.words())
        tree = str(self.tree())
        return f"<OntonotesSentence\n{words}\n{tree}>"
