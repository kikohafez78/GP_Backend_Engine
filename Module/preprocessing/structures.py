import re
import os
import sys
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)



DEFAULT_SENTENCE_BOUNDARIES = [r'(?<=[0-9]|[^0-9.])(\.)(?=[^0-9.]|[^0-9.]|[\s]|$)(?![\n\r]+)',
                               r'\.{2,}', r'\!+', r'\:+', r'\?+', r'[\n\r]+']
DEFAULT_PUNCTUATIONS = [r'(?<=[0-9]|[^0-9.])(\.)(?=[^0-9.]|[^0-9.]|[\s]|$)',
                        r'\.{2,}', r'\!+', r'\:+', r'\?+', r'\,+', r'\(|\)|\[|\]|\{|\}|\<|\>', r'\r\n|\r|\n']


class Document:
 

    def __init__(self, document_text):
 
        self.raw = document_text
        self.sentences = sentencize(self.raw)
        self._index = 0

    def __getitem__(self, key):
        return self.sentences[key]

    def __repr__(self):
        return self.raw

    def __str__(self):
        return self.raw

    def __iter__(self):
        return self

    def __next__(self):
        if self._index < len(self.sentences):
            result = self.sentences[self._index]
            self._index += 1
            return result
        raise StopIteration

    def __len__(self):
        return len(self.sentences)


class Sentence:


    def __init__(self, start_position, end_position, raw_document_reference):
       

        self.start_pos = int(start_position)
        self.end_pos = int(end_position)
        self._document_string = raw_document_reference
        self.next_sentence = None
        self.previous_sentence = None
        self.tokens = tokenize(self._document_string[self.start_pos:self.end_pos])
        self._index = 0

    def get(self):
        return self._document_string[self.start_pos:self.end_pos]

    def __getitem__(self, key):
        return self.tokens[key]

    def __repr__(self):
        return self.get()

    def __str__(self):
        return self.get()

    def __eq__(self, other):
        return self.get() == other

    def __iter__(self):
        return self

    def __next__(self):
        if self._index < len(self.tokens):
            result = self.tokens[self._index]
            self._index += 1
            return result
        raise StopIteration

    def __len__(self):
        return len(self.tokens)


class Token:

    def __init__(self, start_position, end_position, raw_sentence_reference, SOS=False, EOS=False):


        self.start_pos = int(start_position)
        self.end_pos = int(end_position)
        self._sentence_string = raw_sentence_reference
        self.next_token = None
        self.previous_token = None
        self.SOS = SOS
        self.EOS = EOS
        self.PoS = None
        self.raw = self._sentence_string[self.start_pos:self.end_pos]
        self.repr = self.raw

    def get(self):
        if self.SOS:
            return '<SOS>'
        elif self.EOS:
            return '<EOS>'
        else:
            return self.repr

    def __repr__(self):
        return self.get()

    def __str__(self):
        return self.get()

    def __eq__(self, other):
        return self.get() == other



def sentencize(raw_input_document, sentence_boundaries=DEFAULT_SENTENCE_BOUNDARIES, delimiter_token='<SPLIT>'):
  
    if raw_input_document is None or raw_input_document == '':
        raise AttributeError("Empty document string passed as input. Please, verify your input.")
    working_document = raw_input_document
    punctuation_patterns = sentence_boundaries
    for punct in punctuation_patterns:
        working_document = re.sub(punct, r'\g<0>' + delimiter_token, working_document, flags=re.UNICODE)
    list_of_string_sentences = [x.strip(" ") for x in working_document.split(delimiter_token) if x.strip(" ") != ""]
    list_of_sentences = []
    previous = None
    for sent in list_of_string_sentences:
        start_pos = raw_input_document.find(sent)
        end_pos = start_pos + len(sent)
        new_sentence = Sentence(start_pos, end_pos, raw_input_document)
        list_of_sentences.append(new_sentence)
        if previous == None:
            previous = new_sentence
        else:
            previous.next_sentence = new_sentence
            new_sentence.previous_sentence = previous
            previous = new_sentence
    return list_of_sentences


def tokenize(raw_input_sentence, join_split_text=True, split_text_char=r'\-', punctuation_patterns=DEFAULT_PUNCTUATIONS, split_characters=r'[ \t]+', delimiter_token='<SPLIT>'):
    
    if raw_input_sentence is None or raw_input_sentence == '':
        raise None
    working_sentence = raw_input_sentence
    if join_split_text:
        working_sentence = re.sub(r'[a-z]+(' + split_text_char + r'[\n])[a-z]+', '', working_sentence)
    for punct in punctuation_patterns:
        working_sentence = re.sub(punct, r" \g<0> ", working_sentence)
    working_sentence = re.sub(split_characters, delimiter_token, working_sentence)
    list_of_token_strings = [x.strip(" ") for x in working_sentence.split(delimiter_token) if x.strip(" ") != ""]
    previous = Token(0, 0, raw_input_sentence, SOS=True)
    list_of_tokens = [previous]
    for token in list_of_token_strings:
        start_pos = raw_input_sentence.find(token)
        end_pos = start_pos + len(token)
        new_token = Token(start_pos, end_pos, raw_input_sentence)
        list_of_tokens.append(new_token)
        previous.next_token = new_token
        new_token.previous_token = previous
        previous = new_token
    if previous.SOS != True:
        eos = Token(len(raw_input_sentence), len(raw_input_sentence), raw_input_sentence, EOS=True)
        previous.next_token = eos
        eos.previous_token = previous
        list_of_tokens.append(eos)
    return list_of_tokens


def untokenize(token_list):
    
    if not isinstance(token_list, list):
        raise "the entered token list is invalid"
    if len(token_list) < 1:
        return ""
    if not isinstance(token_list[0], Token):
        raise "the entered list does not contain tokens"
    #pointers 
    startpos = 0
    endpos = len(token_list)
    
    if len(token_list) < 3:
        return (" ".join([token.get() for token in token_list])).strip(" ")
    
    if token_list[0] == "<SOS>":
        startpos = 1
    if token_list[-1] == "<EOS>":
        endpos = -1
    punct = "!:?.;,\n"
    final_string = ""
    
    for token in token_list[startpos:endpos]:
        if not token.get() in punct:
            final_string += " "
        final_string += token.get()
        
    return final_string.strip(" ")
