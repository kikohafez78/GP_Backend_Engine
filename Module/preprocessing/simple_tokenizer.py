import string
import re
import os
import sys
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)


DEFAULT_SENTENCE_BOUNDARIES = ['(?<=[0-9]|[^0-9.])(\.)(?=[^0-9.]|[^0-9.]|[\s]|$)','\.{2,}','\!+','\:+','\?+']
DEFAULT_PUNCTUATIONS = ['(?<=[0-9]|[^0-9.])(\.)(?=[^0-9.]|[^0-9.]|[\s]|$)','\.{2,}',
                        '\!+','\:+','\?+','\,+', r'\(|\)|\[|\]|\{|\}|\<|\>']
#simple tokenizer
class sentencizer:
    def __init__(self, sentences: str, split_characters: str, delimieter: str):
        self._split_characters = split_characters
        self.raw = sentences
        self._delimiter_token = delimieter
    def _sentencize(self):
        work_sentence = self.raw
        for character in self._split_characters:
            work_sentence = work_sentence.replace(character, character+""+self._delimiter_token)
        self.sentences = [x.strip() for x in work_sentence.split(self._delimiter_token) if x !='']
    def __iter__(self):
        return self
    def __next__(self):
        if self._index < len(self.sentences):
            result = self.sentences[self._index]
            self._index+=1
            return result
        raise StopIteration
    
class tokenizer:
    def __init__(self, sentence, token_boundaries=[' ', '-'], 
        punctuations=string.punctuation, delimiter_token='<SPLIT>'):
        self.tokens = []
        self.raw = str(sentence)
        self._token_boundaries = token_boundaries
        self._delimiter_token = delimiter_token
        self._punctuations = punctuations
        self._index = 0
        self._tokenize()
    def _tokenize(self):
        work_sentence = self.raw
        for punctuation in self._punctuations:
            work_sentence = work_sentence.replace(punctuation,
                " "+punctuation+" ")
        for delimiter in self._token_boundaries:
            work_sentence = work_sentence.replace(delimiter,
                self._delimiter_token)
        self.tokens = [x.strip() for x in work_sentence.split(self._delimiter_token) if x != '']
    def __iter__(self):
        return self
    def __next__(self):
        if self._index < len(self.tokens):
            result = self.tokens[self._index]
            self._index+=1
            return result
        raise StopIteration
class Document:
    def __init__(self, document_text, sentencize):
        self.raw = document_text
        self.sentences = sentencize(self.raw)
        self._index = 0


class Sentence:
    def __init__(self, start_position, end_position, raw_document_reference, tokenize):
        self.start_pos = int(start_position)
        self.end_pos = int(end_position)
        self._document_string = raw_document_reference
        self.next_sentence = None
        self.previous_sentence = None
        self.tokens = tokenize(self._document_string[self.start_pos:self.end_pos])
        self._index = 0

class Token:
    def __init__(self, start_position, end_position, raw_sentence_reference, SOS = False, EOS = False):
        self.start_pos = int(start_position)
        self.end_pos = int(end_position)
        self._sentence_string = raw_sentence_reference
        self.next_token = None
        self.previous_token = None
        self.SOS = SOS
        self.EOS = EOS

def sentencize(raw_input_document, sentence_boundaries = DEFAULT_SENTENCE_BOUNDARIES, delimiter_token='<SPLIT>'):
    working_document = raw_input_document
    punctuation_patterns = sentence_boundaries
    for punct in punctuation_patterns:
        working_document = re.sub(punct, '\g<0>'+delimiter_token, working_document, flags=re.UNICODE)
    list_of_string_sentences = [x.strip() for x in working_document.split(delimiter_token) if x.strip() != ""]
    list_of_sentences = []
    previous = None
    for sent in list_of_string_sentences:
        start_pos = raw_input_document.find(sent)
        end_pos = start_pos+len(sent)
        new_sentence = Sentence(start_pos, end_pos, raw_input_document)
        list_of_sentences.append(new_sentence)
        if previous == None:
            previous = new_sentence
        else:
            previous.next_sentence = new_sentence
            new_sentence.previous_sentence = previous
            previous = new_sentence
    return list_of_sentences

def tokenize(raw_input_sentence, join_split_text = True, split_text_char = '\-', punctuation_patterns= DEFAULT_PUNCTUATIONS, split_characters = r'\s|\t|\n|\r', delimiter_token='<SPLIT>'):
    working_sentence = raw_input_sentence
    if join_split_text:
        working_sentence = re.sub('[a-z]+('+split_text_char+'[\n])[a-z]+','', working_sentence)
    for punct in punctuation_patterns:
        working_sentence = re.sub(punct, " \g<0> ", working_sentence)
    working_sentence = re.sub(split_characters, delimiter_token, working_sentence)
    list_of_token_strings = [x.strip() for x in working_sentence.split(delimiter_token) if x.strip() !=""]
    previous = Token(0,0,raw_input_sentence, SOS=True)
    list_of_tokens = [previous]
    for token in list_of_token_strings:
        start_pos = raw_input_sentence.find(token)
        end_pos = start_pos+len(token)
        new_token = Token(start_pos,end_pos,raw_input_sentence)
        list_of_tokens.append(new_token)
        previous.next_token=new_token
        new_token.previous_token=previous
        previous=new_token
    if previous.SOS != True:
        eos = Token(len(raw_input_sentence), len(raw_input_sentence), raw_input_sentence, EOS=True)
        previous.next_token=eos
        eos.previous_token = previous
        list_of_tokens.append(eos)
    return list_of_tokens

def seperator(data):
    for line in data:
        res = {'Document': line, 'Sentences':[]}
        document = Document(line)
        for sentence in document:
            res['Sentences'].append({'Sentence':sentence, 'Tokens':sentence.tokens})
    return res