import pickle
import nltk
import re
from spacy.tokens import Doc
import os
import sys
import spacy
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)

class CRFModel:

    nlp = spacy.load("en_core_web_sm")

    def __init__(self, model):
        self.model = model

    def word2features(self, sent, i):
        features = {}
        features.update({'U[0]':'%s' % sent[0][i][0],
                        'POS_U[0]':'%s' % sent[0][i][1]})
        if i < len(sent[0])-1:
            features.update({'U[+1]':'%s' % (sent[0][i+1][0]),
                            'B[0]':'%s %s' % (sent[0][i][0], sent[0][i+1][0]),
                            'POS_U[1]':'%s' % sent[0][i+1][1],
                            'POS_B[0]':'%s %s' % (sent[0][i][1], sent[0][i+1][1])})
            if i < len(sent[0])-2:
                features.update({'U[+2]':'%s' % (sent[0][i+2][0]),
                                'POS_U[+2]':'%s' % (sent[0][i+2][1]),
                                'POS_B[+1]':'%s %s' % (sent[0][i+1][1], sent[0][i+2][1]),
                                'POS_T[0]':'%s %s %s' % (sent[0][i][1], sent[0][i+1][1], sent[0][i+2][1])})
        if i > 0:
            features.update({'U[-1]':'%s' % (sent[0][i-1][0]),
                            'B[-1]':'%s %s' % (sent[0][i-1][0], sent[0][i][0]),
                            'POS_U[-1]':'%s' % (sent[0][i-1][1]),
                            'POS_B[-1]':'%s %s' % (sent[0][i-1][1], sent[0][i][1])})
            if i < len(sent[0])-1:
                features.update({'POS_T[-1]':'%s %s %s' % (sent[0][i-1][1], sent[0][i][1], sent[0][i+1][1])})
            if i > 1:
                features.update({'U[-2]':'%s' % (sent[0][i-2][0]),
                                'POS_U[-2]':'%s' % (sent[0][i-2][1]),
                                'POS_B[-2]':'%s %s' % (sent[0][i-2][1], sent[0][i-1][1]),
                                'POS_T[-2]':'%s %s %s' % (sent[0][i-2][1], sent[0][i-1][1], sent[0][i][1])})
                
        features.update({'INTENT':'%s' % sent[1]})

        return features

    def sent2features(self, sent):
        return [self.word2features(sent, i) for i in range(len(sent[0]))]
    
    def sent2tokens(self, sent):
        return [token for token, _ in sent[0]]

    def predict(self, sents):

        def NLTK_Tokenizer(sentence):
            # Split the sentence into tokens using a regular expression that matches whitespace or punctuation
            sentence = re.sub(r"(?<= )'(.*?)'", r'"\1"', sentence)
            sentence = re.sub(r"\bvs\b\.", r'vs', sentence)
            nltk_sents = nltk.sent_tokenize(sentence)
            tokens = [token for sent in nltk_sents for token in nltk.word_tokenize(sent)]

            return Doc(self.nlp.vocab, tokens)

        self.nlp.tokenizer = NLTK_Tokenizer

        pos_sents = []

        for sample in sents:

            new_sample = []
    
            doc = self.nlp(sample[0])
    
            for token in doc:
                new_sample.append((token.text, token.tag_))
    
            pos_sents.append((new_sample, sample[1]))

        x_pred = [self.sent2features(s) for s in pos_sents]
        y_pred = self.model.predict(x_pred)

        predictions = []

        for i in range(len(pos_sents)):
            predictions.append([pred for pred in zip(self.sent2tokens(pos_sents[i]), y_pred[i])])

        return predictions


    def save_model(self, filename):
          with open(f"{os.path.join(script_dir,f'CRFModelFiles/{filename}.pkl')}", "wb") as file:
            pickle.dump(self.model, file)

    def load_model(self, filename):
        with open(f"{os.path.join(script_dir,f'CRFModelFiles/{filename}.pkl')}", "rb") as file:
             self.model = pickle.load(file)