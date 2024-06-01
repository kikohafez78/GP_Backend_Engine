import os
from sent2vec.vectorizer import Vectorizer
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk.corpus import stopwords
print(load_dotenv(dotenv_path="data.env"))
dataset_path = os.getenv("dataset_path")

def load_data(file_path):
    data = []
    with open(file_path) as file:
        for line in file:
            data.append(line.split('\',')[0][2:])
    
    return data

def vectorize_sent(sent):
    vectorizer = Vectorizer()
    vectorizer.run(sent)
    return vectorizer.vectors

def tf_idf(sent_train, sent_test=None):
    vectorizer = TfidfVectorizer(analyzer='word', input='content')
    if not sent_test:
        return vectorizer.fit_transform(sent_train).toarray(), vectorizer
    else:
        return vectorizer.fit_transform(sent_train).toarray(), vectorizer.transform(sent_test).toarray(), vectorizer

class Dataset:
    def __init__(self):
        self.data = None
        self.labels = None
        self.data_train = None
        self.data_test = None
        self.labels_train = None
        self.labels_test = None
        self.labels_names = None
        self.vectorizer = None


    def prepare_dataset(self, test_size):
        sents = []
        data = []
        labels = []
        labels_names = {}
        index = 0

        data_train = []
        data_test = []
        labels_train = []
        labels_test = []

        for subdir in os.listdir(dataset_path):
            for filename in os.listdir(os.path.join(dataset_path, subdir)):
                sen = load_data(os.path.join(dataset_path, subdir, filename))
                sents.extend(sen)
                lbls = [index for i in range(len(sen))]
                labels.extend(lbls)
                labels_names[index] = filename.split('.')[0]

                x_train, x_test, y_train, y_test = train_test_split(sen, lbls, test_size = test_size)
                data_train.extend(x_train)
                data_test.extend(x_test)
                labels_train.extend(y_train)
                labels_test.extend(y_test)

                index += 1

        self.data, _ = tf_idf(sents)
        self.labels = labels
        self.labels_names = labels_names
        self.labels_train, self.labels_test = labels_train, labels_test
        self.data_train, self.data_test, self.vectorizer = tf_idf(data_train, data_test)