import pandas as pd
from naive_bayes import naive_bayes
from IVF import svm_
from sklearn.feature_extraction.text import TfidfVectorizer
from classical_config import get_classical_config
import pickle as pkl
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,accuracy_score
from keyword_detection import keyword_extraction_module
from masking import masking
import re
import numpy as np
from catboost import Pool, CatBoostClassifier
import json
import sys
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)


class integrated_model_v_1:
    def __init__(self, config):
        self.config = config
        self.sub_classifiers = {}
        self.base_classifier = None
        self.masker = masking("name", self.config)
        
    def initialize(self):
        for class_ in self.n_subclasses:
            self.sub_classifiers[class_] = [naive_bayes(self.config, class_), "model"]
        self.base_classifier = svm_(self.config["base_model_name"], self.config)
        self.feature_extractor = TfidfVectorizer(max_features=self.config["max_seq_len"])
    
    def mask_prompts(self, x):
        return self.masker.mask_corpus(x)    
        
    def train_classifiers(self):
        data = pd.read_csv(self.config["data_path"])
        print(len(data))
        data: pd.DataFrame = data.dropna(0)
        # print("before masking: ", data.loc[0:10, "prompt"])
        # print("==========================================")
        data["prompt"].apply(lambda x: self.mask_prompts(x))
        # print("after masking: ", data.loc[0:10, "prompt"])
        # print("==========================================")
        train , test = train_test_split(data, test_size = 0.005, train_size = 0.995, random_state = 42)
        text_data = train[self.config["input_column"]]
        sub_labels = train[self.config["sub_class_column"]]
        base_labels = train[self.config["base_class_column"]]
        self.sub_class_labels = sub_labels.unique().tolist()
        self.n_subclasses = base_labels.unique().tolist()
        self.initialize()
        features = self.feature_extractor.fit_transform(text_data)
        self.base_classifier.fit(features, base_labels, self.config["do_eval"])
        for classifier in self.sub_classifiers.keys():
            class_specific_data = data[data["classes"] == classifier]
            self.sub_classifiers[classifier][1] = TfidfVectorizer(max_features=self.config["max_seq_len"]) 
            features = self.sub_classifiers[classifier][1].fit_transform(class_specific_data["prompt"])
            labels = class_specific_data["intent"]
            self.sub_classifiers[classifier][0].fit(features, labels, True)
        _, intent_, class_ = self.predict(test[self.config["input_column"]])
        print("pretraining is done for all models")
        print(f"accuracy for integrated model classes is {accuracy_score(test[self.config['base_class_column']], class_)}")
        print(f"accuracy for integrated model intent is {accuracy_score(test[self.config['sub_class_column']], intent_)}")

    
    def save_model(self):
        try:
            os.mkdir(f"{self.config['model_dir']}")
        except FileExistsError:
            print("Directory already exists")
        with open(f"{self.config['model_dir']}/base_feature_extractor.pkl", "wb") as file:
            pkl.dump(self.feature_extractor, file)
        base_name = str(self.base_classifier.name).replace(" ","")
        with open(f"{self.config['model_dir']}/{base_name}.pkl", "wb") as file:
            pkl.dump(self.base_classifier, file)
        for model in self.sub_classifiers.keys():
            sub_name = str(model).replace(" ","")
            with open(f"{self.config['model_dir']}/{sub_name}.pkl", "wb") as file:
                pkl.dump(self.sub_classifiers[model], file)
                
    def load_model(self):
        data = pd.read_csv(self.config["data_path"])
        data = data.dropna(0)
        base_labels = data[self.config["base_class_column"]]
        self.n_subclasses = base_labels.unique().tolist()
        name = str(self.config['base_model_name']).replace(" ","")
        with open(f"{self.config['model_dir']}/base_feature_extractor.pkl", "rb") as file:
            self.feature_extractor = pkl.load(file)
        print("base feature extractor is loaded succefully")
        with open(f"{self.config['model_dir']}/{name}.pkl", "rb") as file:
            self.base_classifier = pkl.load(file)
            print(f"base model with name {self.base_classifier.name} is loaded succefully")
        for model in self.n_subclasses:
            sub_name = str(model).replace(" ","")
            with open(f"{self.config['model_dir']}/{sub_name}.pkl", "rb") as file:
                self.sub_classifiers[model] = pkl.load(file)
                print(f"sub model with name {self.sub_classifiers[model][0].name} is loaded succefully")
        print("all models are loaded succefully")
        
    def predict(self, X):
        predictions = {}
        intent = []
        class_ = []
        for x in X:
            x = self.masker.mask_corpus(x)
            features = self.feature_extractor.transform([x])
            class_pred = self.base_classifier.predict(features)
            intent_pred = self.sub_classifiers[class_pred[0]][0].predict(self.sub_classifiers[class_pred[0]][1].transform([x]))
            predictions[x] = (intent_pred, class_pred[0])
            intent.append(intent_pred)
            class_.append(class_pred)
        return predictions, intent, class_


# class integrated_model_v_2:
    
#     def __init__(self, config):
#         self.config = config
#         self.sub_classifiers = {}
#         self.base_classifier = None
#         self.masker = masking("masker",  self.config)

#     def initialize(self):
#         for class_ in self.n_subclasses:
#             self.sub_classifiers[class_] = comp_naive_bayes(self.config, class_)
#         self.base_classifier = svm_(self.config["base_model_name"], self.config)
#         self.feature_extractor = keyword_extraction_module("featurizer", max_seq_len = self.config["max_seq_len"])
        

#     def mask_prompts(self, x):
#         return self.masker.mask_corpus(x)    
        
#     def train_classifiers(self):
#         data = pd.read_csv(self.config["data_path"])
#         data: pd.DataFrame = data.dropna(0)
#         # print("before masking: ", data.loc[0:10, "prompt"])
#         # print("==========================================")
#         data["prompt"].apply(lambda x: self.mask_prompts(x))
#         # print("after masking: ", data.loc[0:10, "prompt"])
#         # print("==========================================")
#         print(data.shape)
#         train , test = train_test_split(data, test_size = 0.02, train_size = 0.98, random_state = 42)
#         text_data = train[self.config["input_column"]]
#         sub_labels = train[self.config["sub_class_column"]]
#         base_labels = train[self.config["base_class_column"]]
#         self.sub_class_labels = sub_labels.unique().tolist()
#         self.n_subclasses = base_labels.unique().tolist()
#         self.initialize()
#         features = self.feature_extractor.train_svm_data(text_data.to_list())
#         self.base_classifier.fit(features, base_labels, self.config["do_eval"])
#         for classifier in self.sub_classifiers.keys():
#             class_specific_data = data[data["classes"] == classifier]
#             features = self.feature_extractor.train_nb_data(classifier, class_specific_data["prompt"].to_list())
#             labels = class_specific_data["intent"]
#             self.sub_classifiers[classifier].fit(features, labels, True)
#         _, intent_, class_ = self.predict(test[self.config["input_column"]])
#         print("pretraining is done for all models")
#         print(f"accuracy for integrated model classes is {accuracy_score(test[self.config['base_class_column']], class_)}")
#         print(f"accuracy for integrated model intent is {accuracy_score(test[self.config['sub_class_column']], intent_)}")

    
#     def save_model(self):
#         try:
#             os.mkdir(f"{self.config['model_dir']}")
#         except FileExistsError:
#             print("Directory already exists")
#         with open(f"./{self.config['model_dir']}/base_feature_extractor.pkl", "wb") as file:
#             pkl.dump(self.feature_extractor, file)
#         base_name = str(self.base_classifier.name).replace(" ","")
#         with open(f"./{self.config['model_dir']}/{base_name}.pkl", "wb") as file:
#             pkl.dump(self.base_classifier, file)
#         for model in self.sub_classifiers.keys():
#             sub_name = str(model).replace(" ","")
#             with open(f"./{self.config['model_dir']}/{sub_name}.pkl", "wb") as file:
#                 pkl.dump(self.sub_classifiers[model], file)
                
#     def load_model(self):
#         data = pd.read_csv(self.config["data_path"])
#         data = data.dropna(0)
#         base_labels = data[self.config["base_class_column"]]
#         self.n_subclasses = base_labels.unique().tolist()
#         name = str(self.config['base_model_name']).replace(" ","")
#         with open(f"./{self.config['model_dir']}/base_feature_extractor.pkl", "rb") as file:
#             self.feature_extractor = pkl.load(file)
#         print("base feature extractor is loaded succefully")
#         with open(f"{self.config['model_dir']}/{name}.pkl", "rb") as file:
#             self.base_classifier = pkl.load(file)
#             print(f"base model with name {self.base_classifier.name} is loaded succefully")
#         for model in self.n_subclasses:
#             sub_name = str(model).replace(" ","")
#             with open(f"{self.config['model_dir']}/{sub_name}.pkl", "rb") as file:
#                 self.sub_classifiers[model] = pkl.load(file)
#                 print(f"sub model with name {self.sub_classifiers[model][0].name} is loaded succefully")
#         print("all models are loaded succefully")
        
#     def predict(self, X):
#         predictions = {}
#         intent = []
#         class_ = []
#         for x in X:
#             features = self.feature_extractor.encode_svm_([x])
#             class_pred = self.base_classifier.predict(features)
#             intent_pred = self.sub_classifiers[class_pred[0]].predict(self.feature_extractor.encode_nb_(class_pred[0], [x]))
#             predictions[x] = (intent_pred, class_pred[0])
#             intent.append(intent_pred)
#             class_.append(class_pred)
#         return predictions, intent, class_
        
class integrated_model_v_3:
    
    def __init__(self, config):
        self.config = config
        self.sub_classifiers = {}
        self.base_classifier = None
        self.masker = masking("masker",  self.config)

    def initialize(self):
        for class_ in self.n_subclasses:
            self.sub_classifiers[class_] = svm_(class_, self.config)
        self.base_classifier = svm_(self.config["base_model_name"], self.config)
        self.feature_extractor = keyword_extraction_module("featurizer", max_seq_len = self.config["max_seq_len"])
        

    def mask_prompts(self, x):
        return self.masker.mask_corpus(x)    
        
    def train_classifiers(self):
        data = pd.read_csv(self.config["data_path"])
        data: pd.DataFrame = data.dropna(0)
        # print("before masking: ", data.loc[0:10, "prompt"])
        # print("==========================================")
        data["prompt"].apply(lambda x: self.mask_prompts(x))
        # print("after masking: ", data.loc[0:10, "prompt"])
        # print("==========================================")
        train , test = train_test_split(data, test_size = 0.001, train_size = 0.999, random_state = 42)
        text_data = train[self.config["input_column"]]
        sub_labels = train[self.config["sub_class_column"]]
        base_labels = train[self.config["base_class_column"]]
        self.sub_class_labels = sub_labels.unique().tolist()
        self.n_subclasses = base_labels.unique().tolist()
        self.initialize()
        features = self.feature_extractor.train_svm_data(text_data.to_list())
        self.base_classifier.fit(features, base_labels, self.config["do_eval"])
        for classifier in self.sub_classifiers.keys():
            class_specific_data = data[data["classes"] == classifier]
            features = self.feature_extractor.train_nb_data(classifier, class_specific_data["prompt"].to_list())
            labels = class_specific_data["intent"]
            self.sub_classifiers[classifier].fit(features, labels, True)
        _, intent_, class_ = self.predict(test[self.config["input_column"]])
        print("pretraining is done for all models")
        print(f"accuracy for integrated model classes is {accuracy_score(test[self.config['base_class_column']], class_)}")
        print(f"accuracy for integrated model intent is {accuracy_score(test[self.config['sub_class_column']], intent_)}")

    
    def save_model(self):
        try:
            os.mkdir(f"{self.config['model_dir']}")
        except FileExistsError:
            print("Directory already exists")
        with open(f"./{self.config['model_dir']}/base_feature_extractor.pkl", "wb") as file:
            pkl.dump(self.feature_extractor, file)
        base_name = str(self.base_classifier.name).replace(" ","")
        with open(f"./{self.config['model_dir']}/{base_name}.pkl", "wb") as file:
            pkl.dump(self.base_classifier, file)
        for model in self.sub_classifiers.keys():
            sub_name = str(model).replace(" ","")
            with open(f"./{self.config['model_dir']}/{sub_name}.pkl", "wb") as file:
                pkl.dump(self.sub_classifiers[model], file)
                
    def load_model(self):
        data = pd.read_csv(self.config["data_path"])
        data = data.dropna()
        base_labels = data[self.config["base_class_column"]]
        self.n_subclasses = base_labels.unique().tolist()
        name = str(self.config['base_model_name']).replace(" ","")
        with open(f"./{self.config['model_dir']}/base_feature_extractor.pkl", "rb") as file:
            self.feature_extractor = pkl.load(file)
        print("base feature extractor is loaded succefully")
        with open(f"{self.config['model_dir']}/{name}.pkl", "rb") as file:
            self.base_classifier = pkl.load(file)
            print(f"base model with name {self.base_classifier.name} is loaded succefully")
        for model in self.n_subclasses:
            sub_name = str(model).replace(" ","")
            with open(f"{self.config['model_dir']}/{sub_name}.pkl", "rb") as file:
                self.sub_classifiers[model] = pkl.load(file)
                print(f"sub model with name {self.sub_classifiers[model][0].name} is loaded succefully")
        print("all models are loaded succefully")
        
    def predict(self, X):
        predictions = {}
        intent = []
        class_ = []
        for x in X:
            features = self.feature_extractor.encode_svm_([x])
            class_pred = self.base_classifier.predict(features)
            intent_pred = self.sub_classifiers[class_pred[0]].predict(self.feature_extractor.encode_nb_(class_pred[0], [x]))
            predictions[x] = (intent_pred, class_pred[0])
            intent.append(intent_pred)
            class_.append(class_pred)
        return predictions, intent, class_
        
        
        
class integrated_model_v_4:
    
    def __init__(self, config):
        self.config = config
        self.sub_classifiers = {}
        self.base_classifier = None
        self.masker = masking("masker",  self.config)

    def initialize(self):
        self.sub_classifier = CatBoostClassifier(**self.config["catboost_parameters"],class_weights = self.weights)
        self.base_classifier = svm_(self.config["base_model_name"], self.config)
        self.feature_extractor = keyword_extraction_module("featurizer", max_seq_len = self.config["max_seq_len"], config = self.config)
    
    
    def class_weights(self, x, labels):
        cls = np.bincount(x)
        weights = {}
        total = np.sum(cls)
        for label in range(len(labels)):
            weights[label] = (1 / cls[label]) * (total / 2)
        return weights
        
    def remove_emails(self, x):
     return re.sub(r'([a-z0-9+._-]+@[a-z0-9+._-]+\.[a-z0-9+_-]+)',"", x)


    def remove_urls(self, x):
        return re.sub(r'(http|https|ftp|ssh)://([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?', '' , x)

    def remove_rt(self, x):
        return re.sub(r'\brt\b', '', x).strip()

    def remove_special_chars(self, x):
        x = re.sub(r'[^\w ]+', "", x)
        x = ' '.join(x.split())
        return x
    
    def mask_prompts(self, x):
        return self.masker.mask_corpus(x)    
    
    def convert_to_numeric(self, Y, labels: list):
        return labels.index(Y)
    
    # def remove_stopwords(self, x):
    #     return ' '.join([t for t in x.split() if t not in stopwords])
    
    def train_classifiers(self):
        data = pd.read_csv(self.config["data_path"])
        data: pd.DataFrame = data.dropna(0)
        # print("before masking: ", data.loc[0:10, "prompt"])
        # print("==========================================") 
        self.sub_class_labels = data.intent.unique().tolist()
        self.n_subclasses = data.classes.unique().tolist()
        data["prompt"].apply(lambda x: self.mask_prompts(x))
        data['prompt'] = data['prompt'].apply(lambda x: self.remove_emails(x))
        data['prompt'] = data['prompt'].apply(lambda x: self.remove_urls(x))
        data['prompt'] = data['prompt'].apply(lambda x: self.remove_rt(x))
        data['prompt'] = data['prompt'].apply(lambda x: self.remove_special_chars(x))
        data['intent'] = data['intent'].apply(lambda x: self.convert_to_numeric(x, self.sub_class_labels))
        data['classes'] = data['classes'].apply(lambda x: self.convert_to_numeric(x, self.n_subclasses))
        self.weights = self.class_weights(data["intent"], self.sub_class_labels)
        # print(self.weights)
        # print("after masking: ", data.loc[0:10, "prompt"])
        # print("==========================================")
        # print("len of prompts =  %d and len of intent = %d and len of classes = %d",len(data["prompt"]),len(data["intent"]), len(data["classes"]))
        X_train, X_test, Y_train, Y_test = train_test_split(data.drop('intent', axis=1), data["intent"],test_size=0.05,random_state=52)
        # print(f"len of train ({len(X_train)},{len(Y_train)}) and len of test ({len(X_test)},{len(Y_test)})")
        # print(Y_test)
        text_data = data[self.config["input_column"]]
        sub_labels = data[self.config["sub_class_column"]]
        base_labels = data[self.config["base_class_column"]]
        self.sub_class_labels = sub_labels.unique().tolist()
        self.n_subclasses = base_labels.unique().tolist()
        self.initialize()
        features = self.feature_extractor.train_svm_data(text_data.to_list())
        self.base_classifier.fit(features, base_labels, self.config["do_eval"])
        self.train_pool = Pool(
            data = X_train, 
            label = Y_train, 
            cat_features=self.config["cat_features"], 
            text_features=self.config["text_features"], 
            feature_names=list(X_train)
        )
        self.valid_pool = Pool(
            data = X_test, 
            label = Y_test,
            cat_features=self.config["cat_features"], 
            text_features=self.config["text_features"], 
            feature_names=list(X_train)
        )
        self.sub_classifier.fit(self.train_pool, eval_set=self.valid_pool)        
        print("pretraining is done for all models")
        print("starting_evaluation.....")
        # report, accuracy = self.evaluate()
        

    def evaluate(self, X_test, y_test):
        dictionary , pred = self.sub_prediction(X_test)
        report = (classification_report(y_test,pred))
        accuracy = (accuracy_score(y_test,pred))
        return report, accuracy
    
    def save_model(self):
        try:
            os.mkdir(f"{self.config['model_dir']}")
        except FileExistsError:
            print("Directory already exists")
        with open(f"{os.path.join(self.config['model_dir'], 'base_feature_extractor.pkl')}", "wb") as file:
            pkl.dump(self.feature_extractor, file)
        base_name = str(self.base_classifier.name).replace(" ","")
        with open(f"{os.path.join(self.config['model_dir'], f'{base_name}.pkl')}", "wb") as file:
            pkl.dump(self.base_classifier, file)
        with open(f"{os.path.join(self.config['model_dir'], 'class_weights.json')}", "w") as f:
            json.dump(self.weights, f)
        self.sub_classifier.save_model(f"{self.config['sub_model']}",'cbm',None, self.train_pool)
                
    def load_model(self):
        data = pd.read_csv(self.config["data_path"])
        data = data.dropna()
        self.sub_class_labels = data.intent.unique().tolist()
        self.n_subclasses = data.classes.unique().tolist()
        name = str(self.config['base_model_name']).replace(" ","")
        with open(f"{os.path.join(self.config['model_dir'], 'base_feature_extractor.pkl')}", "rb") as file:
            self.feature_extractor = pkl.load(file)
        print("base feature extractor is loaded succefully")
        with open(f"{os.path.join(self.config['model_dir'], f'{name}.pkl')}", "rb") as file:
            self.base_classifier = pkl.load(file)
            print(f"base model with name {self.base_classifier.name} is loaded successfully")
        with open(f"{os.path.join(self.config['model_dir'], 'class_weights.json')}", "r") as f:
            self.weights = json.load(f)
        self.sub_classifier = CatBoostClassifier(**self.config["catboost_parameters"], class_weights = self.weights)
        self.sub_classifier.load_model(self.config["sub_model"])
        print(f"sub model with name {self.config['sub_model']} is loaded successfully")
        
    def predict(self, X):
        predictions = {}
        intent = []
        class_ = []
        for x in X:
            features = self.feature_extractor.encode_svm_([x])
            class_pred = self.base_classifier.predict(features)
            datapoint = pd.DataFrame([(x,class_pred[0])], columns = ["prompt","classes"])
            intent_pred = self.sub_classifier.predict(datapoint)
            predictions[x] = (self.n_subclasses[class_pred[0]], self.sub_class_labels[intent_pred[0][0]])
            intent.append(self.sub_class_labels[intent_pred[0][0]])
            class_.append(self.n_subclasses[class_pred[0]])
        return predictions, intent, class_
    
    def sub_prediction(self, X):
        predictions = {}
        intent = []
        for x in X:
            features = self.feature_extractor.encode_svm_([x])
            class_pred = self.base_classifier.predict(features)
            intent_pred = self.sub_classifier.predict((x,class_pred[0]))
            predictions[x] = intent_pred
            intent.append(intent_pred)
        return predictions, intent
    
    def base_prediction(self, X):
        predictions = {}
        class_ = []
        for x in X:
            features = self.feature_extractor.encode_svm_([x])
            class_pred = self.base_classifier.predict(features)
            predictions[x] = class_pred[0]
            class_.append(class_pred[0])
        return predictions, class_
        
        
    
        
    
# config = get_classical_config()
# model = integrated_model_v_4(config)
# model.train_classifiers()
# model.load_model()
# print(model.predict(["create a new sheet called 'hero'", "create a pivot table"]))