import json
import os
import re
import string
import warnings

import pkg_resources
from symspellpy.symspellpy import SymSpell, Verbosity

from .structures import sentencize, tokenize, Sentence, Document, untokenize
import sys
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)

contractions_file = os.path.join(os.path.dirname(__file__),"./preloaded/dictionaries/contractions/english_contractions.json")
stopwords_file = os.path.join(os.path.dirname(__file__), "./preloaded/lists/words/english_stopwords.txt")



def simplify_punctuation(text):
    if text is None:
        return ''
    corrected = str(text)
    corrected = re.sub(r'([!?,;])\1+', r'\1', corrected)
    corrected = re.sub(r'\.{2,}', r'...', corrected)
    return corrected


def normalize_whitespace(input_string):
    if input_string is None:
        return ''
    corrected = str(input_string)
    corrected = re.sub(r"//t", r"\t", corrected)
    corrected = re.sub(r"( )\1+", r"\1", corrected)
    corrected = re.sub(r"(\n)\1+", r"\1", corrected)
    corrected = re.sub(r"(\r)\1+", r"\1", corrected)
    corrected = re.sub(r"(\t)\1+", r"\1", corrected)
    return corrected.strip(" ")


def normalize_contractions(token_list):
    contractions = json.loads(open(contractions_file, 'r').read())
    new_token_list = []
    for word_pos in range(len(token_list)):
        word_token = token_list[word_pos]
        word = word_token
        first_upper = False
        if word[0].isupper():
            first_upper = True
        if word.lower() in contractions:
            replacement = contractions[word.lower()]
            if first_upper:
                replacement = replacement[0].upper() + replacement[1:]
            replacement_tokens = replacement.split()
            if len(replacement_tokens) > 1:
                new_token_list.append(replacement_tokens[0])
                new_token_list.append(replacement_tokens[1])
            else:
                new_token_list.append(replacement_tokens[0])
        else:
            new_token_list.append(word_token)
    tokens = tokenize(" ".join(new_token_list).strip(" "))
    return tokens


def remove_stopwords(token_list):
    stopwords = [line.replace('\n', '') for line in open(stopwords_file, 'r').readlines()]
    for word_pos in range(1, len(token_list[:-1])):
        word_token = token_list[word_pos]
        word = word_token.get()
        if word in stopwords:
            word_token.repr = ""
            token_list[word_pos] = word_token
    return token_list


def spell_correction(token_list):
    spellchecker = SysmspellSingleton()
    max_edit_distance_lookup = 2
    suggestion_verbosity = Verbosity.TOP
    for word_pos in range(1, len(token_list[:-1])):
        word_token = token_list[word_pos]
        word = word_token.get()
        if not '\n' in word and word not in string.punctuation and not is_numeric(word) and not (word.lower() in spellchecker.words.keys()):
            suggestions = spellchecker.lookup(word.lower(), suggestion_verbosity, max_edit_distance_lookup)
            upperfirst = word[0].isupper()
            if len(suggestions) > 0:
                correction = suggestions[0].term
                replacement = correction
            else:
                replacement = _reduce_exaggerations(word)
            if upperfirst:
                replacement = replacement[0].upper() + replacement[1:]
            word_token.repr = replacement
            token_list[word_pos] = word_token
    return token_list


def _reduce_exaggerations(text):
    correction = str(text)
    return re.sub(r'([\w])\1+', r'\1', correction)


def is_numeric(text):
    for char in text:
        if not (char in "0123456789" or char in ",%.$"):
            return False
    return True



class SysmspellSingleton:
    _instance = None
    def __new__(cls):
        if cls._instance is None:
            max_edit_distance_dictionary = 3
            prefix_length = 4
            spellchecker = SymSpell(max_edit_distance_dictionary, prefix_length)
            dictionary_path = pkg_resources.resource_filename(
                "symspellpy", "frequency_dictionary_en_82_765.txt")
            bigram_path = pkg_resources.resource_filename(
                "symspellpy", "frequency_bigramdictionary_en_243_342.txt")
            spellchecker.load_dictionary(dictionary_path, term_index=0, count_index=1)
            spellchecker.load_bigram_dictionary(dictionary_path, term_index=0, count_index=2)
            cls._instance = spellchecker
        return cls._instance


class Normalizer:
    pre_tokenization_functions = {'simplify_punctuation': simplify_punctuation,
                                  'normalize_whitespace': normalize_whitespace}
    post_tokenization_functions = {'normalize_contractions': normalize_contractions,
                                   'spell_correction': spell_correction,
                                   'remove_stopwords': remove_stopwords}

    def __init__(self, pre_tokenization_steps=['simplify_punctuation', 'normalize_whitespace'], post_tokenization_steps=['normalize_contractions', 'spell_correction']):
        self.pre_tokenization_steps = [self.pre_tokenization_functions[step] for step in pre_tokenization_steps]
        self.post_tokenization_steps = [self.post_tokenization_functions[step] for step in post_tokenization_steps]

    def normalize_string(self, input_string):
        if input_string is None:
            return ''
        normalized_string = str(input_string)
        for pre_tokenization_step in self.pre_tokenization_steps:
            normalized_string = pre_tokenization_step(normalized_string)
        tokens = tokenize(normalized_string)
        for post_tokenization_step in self.post_tokenization_steps:
            tokens = post_tokenization_step(tokens)
        return untokenize(tokens)

    def normalize_document(self, input_document):
        warnings.warn(
            "This function can imply in data loss and reverts all lemmatization and pos tagging process. It is recomended that any text is normalized prior to tokenization.")
        if not isinstance(input_document, Document):
            raise TypeError(
                message="Wrong argument provided. Please, ensure that the input is of type core.structures.Document")
        sentences = input_document.sentences
        new_sentences = []
        for sentence in sentences:
            new_sentences.append(self.normalize_string(sentence))
        new_raw_document = " ".join(new_sentences)
        return Document(new_raw_document.strip(" "))

    def normalize_sentence(self, input_sentence):
        warnings.warn("This function can imply in data loss and reverts all lemmatization and pos tagging process. Also, it creates a standalone sentence in relation to parent Document. It is recomended that any text is normalized prior to tokenization.")
        if not isinstance(input_sentence, Sentence):
            raise TypeError(
                message="Wrong argument provided. Please, ensure that the input is of type core.structures.Sentence")
        raw_sentence = input_sentence.get()
        raw_sentence = self.normalize_string(raw_sentence)
        return sentencize(raw_sentence)[0]

    def normalize_raw_document(self, document_path):
        with open(document_path) as f:
            lines = f.read()
        document_to_normalize = Document(lines)
        normalized_document = self.normalize_document(document_to_normalize)

        return normalized_document.raw
