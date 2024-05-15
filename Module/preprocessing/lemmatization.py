import pickle, os

from word_utils import get_inflected_noun

class AbstractLemmatizer:
    def lemmatize():
        pass

class DictionaryLemmatizer(AbstractLemmatizer):

    dict_directory = os.path.join(os.path.dirname(__file__), "../preloaded/dictionaries/lemmas/word_lemma_dict.p")

    def __init__(self):
        self.lemma_dict = pickle.load(open(self.dict_directory,'rb'))

    def lemmatize(self, word, tag, lemmatize_plurals=True):
        if word is None:
            return ''
        if pos is None:
            pos = ''
        word, tag = str(word).lower(), str(tag).upper()
        if word in self.lemma_dict  and pos in self.lemma_dict[word]:
                return self.lemma_dict[word][pos]
        if pos == "NOUN" and lemmatize_plurals:
            return get_inflected_noun(word)
        return word
