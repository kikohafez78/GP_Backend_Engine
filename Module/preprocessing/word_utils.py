import pickle
import os
import sys
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)
import numpy as np

def sigmoid(x):
    return 1/(1 +np.exp(-x))


dict_directory = os.path.join(os.path.dirname(__file__), "./preloaded/dictionaries/lemmas/irregular_noun_dict.p")
words_list_directory = os.path.join(os.path.dirname(__file__), "./preloaded/lists/words/words_alpha.txt")

def get_inflected_noun(word):
    irregular = pickle.load(open(dict_directory,'rb'))
    consonants = "bcdfghjklmnpqrstwxyz"
    vowels = "aeiou"
    word = str(word).lower()
    if len(word) < 2:
        return word
    if word in irregular:
        return irregular[word]
    
    if word.endswith('s'):
        if len(word) > 3:
            if word.endswith('ves'):
                if len(word[:-3]) > 2:
                    return word.replace('ves','f')
                else:
                    return word.replace('ves','fe')
                
            if word.endswith('ies'):
                return word.replace('ies','y')
            
            if word.endswith('es'):
                if word.endswith('ses') and word[-4] in vowels:
                    return word[:-1]
                if word.endswith('zzes'):
                    return word.replace('zzes','z')
                return word[:-2]
            
            if word.endswith('ys'):
                return word.replace('ys','y')
            return word[:-1]
    return word


def levenshtein_distance(word1, word2):
    word1 = str(word1)
    word2 = str(word2)
    comparison_matrix = np.zeros(shape=(len(word1)+1,len(word2)+1))
    comparison_matrix[:,0] = [m for m in range(len(word1)+1)]
    comparison_matrix[0,:] = [n for n in range(len(word2)+1)]
    for x in range(1, len(word1)+1):
        for y in range(1, len(word2)+1):
            if word1[x-1] == word2[y-1]:
                up = comparison_matrix[x-1][y]+1
                upper_diag = comparison_matrix[x-1][y-1]
                left = comparison_matrix[x, y-1]+1
                minimum = min(up, upper_diag, left)
                comparison_matrix[x][y] = minimum
            else:
                up = comparison_matrix[x-1][y]+1
                upper_diag = comparison_matrix[x-1][y-1]+1
                left = comparison_matrix[x, y-1]+1
                minimum = min(up, upper_diag, left)
                comparison_matrix[x][y] = minimum
    return comparison_matrix[-1,-1]


class AbstractSpellCorrector:
    def correct():
        pass

class SimpleSpellCorrector(AbstractSpellCorrector):
 

    def __init__(self):
        word_list = open(words_list_directory,'r').readlines()
        self.words = set(word_list)

    def correct(self, word):
        if word in self.words:
            return word
        else:
            return min(self._candidates(word), key= lambda k: self._word_score(word, k))

    def similarity(self, word1, word2):
        lev_score = levenshtein_distance(word1, word2)
        letters1 = {}
        for letter in word1:
            letters1[letter] = letters1.get(letter, 0)+1
        letters2 = {}
        for letter in word2:
            letters2[letter] = letters2.get(letter, 0)+1
        letter_score = self._dict_compare(letters1, letters2)
        size_dif = abs(len(word1)-len(word2))
        print(lev_score+letter_score+size_dif, word2)
        return lev_score+letter_score+size_dif

    def _dict_compare(self, d1, d2):
        d1_keys = set(d1.keys())
        d2_keys = set(d2.keys())
        intersect_keys = d1_keys.intersection(d2_keys)
        added = d1_keys-d2_keys
        removed = d2_keys-d1_keys
        modified = {o: (d1[o], d2[o]) for o in intersect_keys if d1[o]!=d2[o]}
        return sigmoid(len(added)*2+len(removed)*2+len(modified))

    def _candidates(self, word):
        return (self._existing([word]) or self._existing(self._generate_candidates_1_away(word)) or self._existing(self._generate_candidates_2_away(word)) or [word])

    def _existing(self, words):
        return set(word for word in words if word in self.words)

    def _generate_candidates_1_away(self, word):
        letters = 'abcdefghijklmnopqrstuvwxyz'
        splits     = [(word[:i], word[i:])    for i in range(len(word) + 1)]
        deletes    = [L + R[1:]               for L, R in splits if R]
        transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]
        replaces   = [L + c + R[1:]           for L, R in splits if R for c in letters]
        inserts    = [L + c + R               for L, R in splits for c in letters]
        return set(deletes + transposes + replaces + inserts)

    def _generate_candidates_2_away(self, word):
        return set(e2 for e1 in self._generate_candidates_1_away(word) for e2 in self._generate_candidates_1_away(e1))
