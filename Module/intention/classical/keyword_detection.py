# from ...preprocessing.preprocessing import preprocessing 
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.base import TransformerMixin
import json
import sys
import os
from scipy.sparse import hstack
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)

class keyword_engine(TransformerMixin):
    def __init__(self, keywords: list):
        self.keywords = keywords
            
            
    def fit(self, X, y = None):
        return self
    
    def transform(self, X):
        keyword_features = []
        for keywords in self.keywords:
            keyword_features.append(" ".join(keywords))
        return keyword_features
    
    

class keyword_module:
    class_names = [
        "entry and manipulation",
        "management",
        "formatting",
        "charts",
        "pivot tables",
        # "math functions",
        # "text functions",
        # "financial functions",
        # "array functions",
        # "lookup functions"
    ]
    def __init__(self, name: str, config):
        self.name = name
        lines = []
        with open(f"{os.path.join(config['misc_dir'], 'relations.json')}", "r") as file:
            self.class_relations: dict = json.load(file)
        with open(f"{os.path.join(config['misc_dir'], 'keywords.txt')}", "r") as file:
            lines = file.read()
        self.sentences_dict = {}
        keywords = set()
        self.class_keywords = {class_name: set() for class_name in self.class_relations.keys()}
        for line in lines.strip().split("\n"):
            key, value = line.split(":")
            value = {phrase.replace('"', '').replace('.','').strip().lower() for phrase in value.split(",")}
            keywords = keywords.union(value)
            for class_ in self.class_relations.keys():
                if key in self.class_relations[class_]:
                    self.class_keywords[class_] = self.class_keywords[class_].union(value)
        
        self.keywords = list(keywords)
        for class_ in self.class_keywords.keys():
            self.class_keywords[class_] = list(self.class_keywords[class_]) 
            
        self.svm_count_vectorizer = CountVectorizer(binary = True, vocabulary = self.keywords, ngram_range=(1,3))
        self.nb_pipelines = {}
        for class_ in self.class_names:
            self.nb_pipelines[class_] = CountVectorizer(binary = True, vocabulary = self.class_keywords[class_],ngram_range=(1,3))
        
    def train_svm_vectorizer(self, corpus):
        return self.svm_count_vectorizer.fit_transform(corpus)
    
    def train_nb_vectorizer(self, class_, corpus):
        return self.nb_pipelines[class_].fit_transform(corpus)
    
    def svm_keywords(self, text):
        return self.svm_count_vectorizer.transform(text).toarray()
    
    def nb_keywords(self, class_, text):
        return self.nb_pipelines[class_].transform(text).toarray()
    
    
# import pandas as pd
# from classical_config import get_classical_config

# config = get_classical_config()
# engine = keyword_module("name")
# csv = pd.read_csv("../data/Book1.csv")
# (engine.train_svm_vectorizer(csv["prompt"].to_list()))
# print(engine.svm_keywords(["focus on points"]))


    
class keyword_extraction_module:
    class_names = [
        "entry and manipulation",
        "management",
        "formatting",
        "charts",
        "pivot tables",
        # "math functions",
        # "text functions",
        # "financial functions",
        # "array functions",
        # "lookup functions"
    ]
    
    def __init__(self, name: str, max_seq_len: int, config: dict):
        self.name = name
        self.svm_tfidf = TfidfVectorizer(max_features = max_seq_len)
        self.sub_tfidf = {}
        
        for class_ in self.class_names:
            self.sub_tfidf[class_] = TfidfVectorizer(max_features = max_seq_len)
        self.keyword_extractor_mod = keyword_module("keyword features", config)
        
        
    def train_svm_data(self, corpus):
        tfidf = self.svm_tfidf.fit_transform(corpus)
        keywords = self.keyword_extractor_mod.svm_count_vectorizer.fit_transform(corpus)
        return hstack([tfidf,keywords])
    
    def train_nb_data(self, class_, corpus):
        tfidf = self.sub_tfidf[class_].fit_transform(corpus)
        keywords = self.keyword_extractor_mod.nb_pipelines[class_].fit_transform(corpus)
        return hstack([tfidf,keywords])
        
    def encode_svm_(self, text):
        tfidf = self.svm_tfidf.transform(text)
        keywords = self.keyword_extractor_mod.svm_count_vectorizer.transform(text)
        return hstack([tfidf,keywords])
    
    
    def encode_nb_(self, class_, corpus):
        tfidf = self.sub_tfidf[class_].transform(corpus)
        keywords = self.keyword_extractor_mod.nb_pipelines[class_].transform(corpus)
        return hstack([tfidf,keywords])



# import pandas as pd

# data = pd.read_csv("../data/Book1.csv")
# feature_model = keyword_extraction_module("name", 300)
# feature_model.train_svm_data(data["prompt"].to_list())
# features = feature_model.encode_svm_(["modify a the new sheet and edit its title"])
# print("combined: ",features.shape)