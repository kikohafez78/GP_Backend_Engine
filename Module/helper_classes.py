import os
import sys
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)
from nltk import WordNetLemmatizer
import string 

class document(object):
    def __init__(self, text: str, delimiters: str = "<SPLIT>", splitters: list[str] = ["?", "!", ":", "."], lemmitizer: WordNetLemmatizer = None):
        self.text = str(text)
        self.splitters = splitters
        self.delimiters = delimiters
        self.i = 0
        self.sentencing()
        self.lemmitizer = lemmitizer
        self.stemmer = None
        self.documents - None
        
    def sentencing(self):
        target = self.text
        for splitter in self.splitters:
            target = target.replace(splitter, splitter+""+self.delimiters)
        self.documents = [x.strip() for x in target.split(self.delimiters) if x != '']
        return self.documents    
    
    def __repr__(self):
        return f"the sentence being tokenized is {self.text}"
    def __next__(self):
        if len(self.documents) > self.i:
            results = self.documents[self.i]
            self.i += 1
            return results
        
        
class tokenizer(string):
    def __init__(self, line: str, puncts, delimiters: str = "<SPLIT>", boundries: list[str] = [' ','-']):
        self.text = str(line)
        self.tokens = []
        self.puncts = puncts
        self.bounds = boundries
        self.delimiters = delimiters
        self.i = 0
        self.tokenize()
    def tokenize(self):
        target = self.text
        for delimiter in self.bounds:
            target = target.replace(delimiter, self.delimiters)
        self.tokens = [char.strip() for char in target.split(self.delimiters) if char != '']
    def __next__(self):
        if self.i < len(self.tokens):
            result = self.tokens[self.i]
            self.i+=1
            return result
    def __repr__(self):
        return f"""this tokenizer has the following properties
    delimiter: {self.delimiters}
    sentence boundries: {self.bounds},
    punctuations: {self.puncts}
    """
class Document(object):
    def __init__(self, title: str, text: str):
        self.title  = title
        self.text = text
        self.i = 0
    def __repr__(self):
        return f"""the document title: {self.title}\n
        index: {self.i}\n
        content: {self.text}\n
        """
    def size(self):
        return len(self.text)