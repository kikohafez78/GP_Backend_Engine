import string
from sklearn.naive_bayes import MultinomialNB
from nltk import word_tokenize
import math
from sklearn.metrics import accuracy_score
class NBMC(object): #Naive bayes multi class 
    def __init__(self, classes):
        self.classes = classes
        self.num_classes = len(classes)
    
    
    def get_stopwords_list(self):
        stopwords_list = None
        with open("stopwords-en.txt", "r") as file:
            stopwords_list = file.readlines()
        return stopwords_list
        
    #string cleanup removing punctuaction
    def clean_string(self, s):
        return s.translate(str.maketrans('', '', string.punctuation))

    #tokenize strings into words
    def tokenize_string(self, text):
        return word_tokenize(text)
 
    #count up how many of each word appears in a list of words
    def get_word_counts(self, words):
        word_counts = {}
        for word in words:
            word_counts[word] = word_counts.get(word, 0.0) + 1.0
        return word_counts

    def train(self, X, Y):       
        self.num_texts = {}
        self.log_prior_probabilities = {}
        self.bow = {}
        #global vocabulary
        self.global_vocabulary = set() 
        #list containing english stopwords
        self.stopwords_list = self.get_stopwords_list()
        n = len(X)
        for num_text in self.classes:
            self.num_texts[num_text] = sum(1 for label in Y if label == self.classes.index(num_text))
            self.log_prior_probabilities[num_text] = math.log(self.num_texts[num_text] / n)
            self.bow[num_text] = {}
        #X,Y are iterables
        for x, y in zip(X, Y):
            if isinstance(x, float):
                print(x)
                continue
            c = self.classes[y]
            counts = self.get_word_counts(self.tokenize_string(x))
            for word, count in counts.items():
                #removing stop words
                if (word not in self.stopwords_list):
                    if word not in self.global_vocabulary:
                        self.global_vocabulary.add(word)
                    if word not in self.bow[c]:
                        self.bow[c][word] = 0.0
                    self.bow[c][word] += count

    def predict(self, X):
        result = []
        for x in X:

            counts = self.get_word_counts(self.tokenize_string(x))
            class_scores = [[class_name, 0.0] for class_name in self.classes]
            for word, _ in counts.items():
                if word not in self.global_vocabulary: continue
                for class_ in class_scores:
                    log_w = math.log( (self.bow[class_[0]].get(word, 0.0) + 1) / (self.num_texts[class_[0]] + len(self.global_vocabulary)) )
                    class_[1] += log_w
            for class_ in class_scores:
                class_[1] += self.log_prior_probabilities[class_[0]]
            #compute result
            aux = {class_name:value for class_name, value in class_scores}
            maxValue = max(aux.values())
            print("log linear probabilities: ", aux)
            
            for k,v in aux.items():
                if v == maxValue:
                    result_to_append = k
            print("max: ", maxValue, "class: ", result_to_append)
            result.append(result_to_append)
            
        return result
    
    
class naive_bayes:
    def __init__(self, config, name):
        self.name = name
        self.model = MultinomialNB(alpha = config["alpha"], force_alpha=config["force_alpha"], fit_prior=config["fit_prior"])
        
        
    def fit(self, X, Y, do_eval: bool = True):
        self.model.fit(X,Y)
        predictions = []
        for x in X:
            y_pred = self.model.predict(x)
            predictions.append(y_pred)
        if do_eval:
            print(f"the model {self.name} NB accuracy is {accuracy_score(Y,predictions)}")
            
    
    def predict(self, X):
        return self.model.predict(X)