from structures import Token
import os
import sys
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)



class AbstractStemmer:
    def stem(self, word):
        pass

class PorterStemmer(AbstractStemmer):
    consonants = "bcdfghjklmnpqrstwxz"
    special_case = "y"
    vowels = "aeiou"

    def stem(self, word):

        if isinstance(word, Token):
            word = word.get()
        stem = word.lower().strip()
        stem = self._porter_step_1(stem)
        stem = self._porter_step_2(stem)
        stem = self._porter_step_3(stem)
        stem = self._porter_step_4(stem)
        stem = self._porter_step_5(stem)
        return stem

    def _divide_into_groups(self, word: str):
        groups = []
        preceding = ""
        for idx, letter in enumerate(word.lower()):
            if preceding == "":
                preceding = letter
            else:
                if self._compare_same_class(preceding, letter):
                    preceding+= letter
                    if idx == len(word)-1:
                        groups.append(preceding)
                else:
                    groups.append(preceding)
                    preceding = letter
                    if idx == len(word)-1:
                        groups.append(letter)
        return groups

    def _compare_same_class(self, l1, l2):
        if l1 in self.consonants and l2 in self.consonants:
            return True
        elif l1 in self.vowels and l2 in self.vowels:
            return True
        else:
            return False
        return False

    def _determine_class(self, group):
        if group[0] in self.consonants:
            return 'C'
        return 'V'

    def _encode_word(self, word):
        encoded = self._divide_into_groups(word)
        classified = [self._determine_class(group) for group in encoded]
        return classified

    def _det_m(self, word):
        classes = self._encode_word(word)
        if len(classes) < 2:
            return 0
        if classes[0] == 'C':
            classes = classes[1:]
        if classes[-1] == 'V':
            classes = classes[:len(classes)-1]
        m = len(classes)//2 if (len(classes)/2) >= 1 else 0
        return m

    def _chk_LT(self, stem, lt):
        for letter in lt:
            if stem.endswith(letter):
                return True
        return False

    def _chk_v(self, stem):
        for letter in stem:
            if letter in self.vowels:
                return True
        return False

    def _chk_d(self, stem):
        if stem[-1] in self.consonants and stem[-2] in self.consonants:
            return True
        return False

    def _chk_o(self, stem):
        if len(stem) <3:
            return False
        if (stem[-3] in self.consonants) and (stem[-2] in self.vowels) and (stem[-1] in self.consonants) and (stem[-1] not in "wxy"):
            return True
        else:
            return False

    def _porter_step_1(self, word):
        stem = word
        stepb2 = False
        if stem.endswith('sses'):
            stem = stem[:-2]
        elif stem.endswith('ies'):
            stem = stem[:-2]
        elif not stem.endswith('ss') and stem.endswith("s"):
            stem = stem[:-1]
        if len(stem) > 4:
            if stem.endswith("eed") and self._det_m(stem) > 0:
                stem = stem[:-1]
            elif stem.endswith("ed"):
                stem = stem[:-2]
                if not self._chk_v(stem):
                    stem = word
                else:
                    stepb2 = True
            elif stem.endswith("ing"):
                stem = stem[:-3]
                if not self._chk_v(stem):
                    stem = word
                else:
                    stepb2 = True
        if stepb2:
            if stem.endswith("at") or stem.endswith("bl") or stem.endswith("iz"):
                stem += "e"
            elif self._chk_d(stem) and not (self._chk_LT(stem,"lsz")):
                stem = stem[:-1]
            elif self._det_m(stem)==1 and self._chk_o(stem):
                stem += "e"
        if self._chk_v(stem) and stem.endswith('y'):
            stem = stem[:-1]+'i'
        return stem

    def _porter_step_2(self, stem):
        pair_tests = [('ational','ate'), ('tional','tion'), ('enci','ence'), ('anci','ance'), ('izer', 'ize'),
                      ('abli','able'), ('alli','al'), ('entli', 'ent'), ('eli', 'e'), ('ousli', 'ous'), ('ization', 'ize'),
                      ('ation', 'ate'), ('ator', 'ate'), ('alism', 'al'), ('iveness', 'ive'), ('fulness', 'ful'),
                      ('ousness', 'ous'), ('aliti','al'), ('ivit', 'ive'), ('biliti','ble')]
        if self._det_m(stem) > 0:
            for term, subs in pair_tests:
                if stem.endswith(term):
                    return stem[:-len(term)]+subs
        return stem

    def _porter_step_3(self, stem):
        pair_tests = [('icate','ic'),('ative',''),('alize','al'),('iciti','ic'),('ical','ic'),('ful',''),('ness','')]
        if self._det_m(stem) > 0:
            for term, subs in pair_tests:
                if stem.endswith(term):
                    return stem[:-len(term)]+subs
        return stem

    def _porter_step_4(self, stem):
        suffixes_1 = ['al','ance','ence','er','ic','able','ible','ant','ement','ment','ent']
        special_case = 'ion'
        suffixes_2 = ['ou','ism','ate','iti','ous','ive','ize']
        if self._det_m(stem)>1:
            for suffix in suffixes_1:
                if stem.endswith(suffix):
                    return stem[:-len(suffix)]
            if stem.endswith(special_case):
                temp = stem[:-len(special_case)]
                if self._chk_LT(temp, 'st'):
                    return temp
            for suffix in suffixes_2:
                if stem.endswith(suffix):
                    return stem[:-len(suffix)]
        return stem

    def _porter_step_5(self, stem):
        temp = stem
        if self._det_m(temp)>1 and temp.endswith('e'):
            temp = temp[:-1]
        elif self._det_m(temp) == 1 and (not self._chk_o(temp)) and temp.endswith('e') and len(temp) > 4:
            temp = temp[:-1]
        if self._det_m(temp) > 1 and self._chk_d(temp) and self._chk_LT(temp, 'l'):
            temp = temp[:-1]
        return temp
