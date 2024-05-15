import numpy as np
import logging as logs
from collections import Counter
import openpyxl as excel
import os
import pandas as pd
from helper_functions import load_pkl
from sklearn.feature_extraction.text import TfidfVectorizer

#gets the Term frequency
def TF(word_counts: dict):
    total_count = sum(word_counts.values())
    word_freq = {word:(counts / total_count) for word, counts in word_counts.items()}
    return total_count, word_freq

#gets the inverse document frequency
def IDF(data: list[str]):
    words = {word for doc in data for word in doc}
    idf = {}
    number_of_docs = len(data)
    for word in words:
        c = sum(1 for doc in data if word in doc)
        idf[word] = np.log(number_of_docs / c)
    return idf

#gets the combination of both the functions above it
#low appearing words in documents will have higher importance and weights
def TFIDF(data):
    tfidf = {}
    idf = IDF(data)    
    for num, doc in enumerate(data):
        counts = Counter(doc)        
        tf = TF(counts)
        tfidf[num] = {word:(tf[word] * idf[word]) for word in doc}
    return tfidf



#calculate tfidf given filename
def get_tfidf_given_file(filename):
    if not os.path.exists(filename):
        logs.log(0,"file used doesn't exists in the current workspace ")
        return None
    filename, extension = filename.split(".")
    data = None
    if extension == "pkl":
        data = np.load(filename+"."+extension)
    elif extension == "csv":
        data = pd.read_csv(filename+"."+extension)
    elif extension == "xlsx":
        data = excel.load_workbook(filename+"."+extension)
        sheet_names = data.sheetnames
        sheets = [data[name] for name in sheet_names]
        data = sheets
    elif extension == "pkl":
        data = load_pkl(filename+"."+extension)
        
def tfidf_given_text(corpus):
    vectorizer = TfidfVectorizer()
    vectorizer.fit_transform(corpus)
    return vectorizer