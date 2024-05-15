# -*- coding: utf-8 -*-
"""
A data structure that represents a document.
A document contains multiple Sentence objects and
informations spanning multiple sentences like coreference infos.
"""
import logging
import os

from .sentence import Sentence


class Document:

    def __init__(self, path):
        self._path = path
        self._sentences = self._read_sentences(path)

    @property
    def path(self):
        return self._path

    def filename(self):
        return os.path.split(self._path)[1].split(".")[0]

    def coreference_chains(self):
        """Returns a dictionary that contains coreference information.

        Returns:
            A dictionary where the keys are abitrary ids and
            the values are sets of 3-tuples
            (sentence-index, start, end) representing mentions that are
            coreferential.
        """
        coref_dict = dict()
        for sent in self:
            sent_coref = sent.coreference()
            for coref in sent_coref:
                coref_dict.setdefault(coref, set())
                mentions = ((sent.index, start, end)
                            for start, end in sent_coref[coref])
                coref_dict[coref] = coref_dict[coref].union(mentions)
        return coref_dict

    def named_entities(self):
        """Returns a dictionary that contains named entity information.

        Returns:
            A dictionary where the keys are 3-tuples representing
            a named entity and the values are the named entity type.
        """
        ne_dict = dict()
        for sent in self:
            sent_ne = sent.named_entities()
            for start, end in sent_ne:
                ne_dict[(sent.index, start, end)] = sent_ne[(start, end)]
        return ne_dict

    def _read_sentences(self, path):
        """Read in a conll-document from a file."""
        sentences = []
        tmp_sentence = []
        sent_id = 0
        with open(os.path.join(path), encoding="utf-8") as doc_file:
            for line in doc_file:
                # These lines define start and end of docs.
                if line.startswith("#"):
                    continue
                line = line.rstrip()
                # An empty line means that a sentence has ended.
                if not line:
                    try:
                        sentence = Sentence(sent_id, tmp_sentence)
                    except (IndexError, ValueError) as e:
                        logging.warning(f"Malformed sentence {sent_id} "
                                        "in {path} ignored.")
                        logging.warning(f"{e}")
                        
                        continue

                    else:
                        sentences.append(sentence)
                        sent_id += 1
                    finally:
                        tmp_sentence = []
                else:
                    # Collect lines of a sentence.
                    tmp_sentence.append(line.split())
            # There might still be a sentence left if
            # there is no empty line at the end.
            if tmp_sentence:
                sentence = Sentence(sent_id, tmp_sentence)
                sentences.append(sentence)
        return sentences

    def __getitem__(self, index):
        return self._sentences[index]

    def __iter__(self):
        return iter(self._sentences)
