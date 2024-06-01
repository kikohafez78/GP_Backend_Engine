import os
from dotenv import load_dotenv
from Prepare_Dataset import Dataset
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import pickle
load_dotenv(dotenv_path="data.env")

class knn_tf_idf_model:

    def __init__(self):
        self.model = None
        self.data = Dataset()
        self.test_size = float(os.getenv("test_size"))
        self.data.prepare_dataset(self.test_size)

    def train(self):
        knn = KNeighborsClassifier(n_neighbors = int(os.getenv("n_neighbors")), metric= 'cosine')

        knn.fit(self.data.data_train, self.data.labels_train)

        labels_pred = knn.predict(self.data.data_test)

        self.model = knn

        accuracy = accuracy_score(self.data.labels_test, labels_pred)
        print("Accuracy:", accuracy)

    def save_model(self, filename):
        model_pkl = open(filename, 'wb')
        pickle.dump(self.model, model_pkl)
        model_pkl.close()

    def load_model(self, filename):
        self.model = pickle.load(open(filename, 'rb'))

    def predict(self, sentences: list):
        sent_vecs = self.data.vectorizer.transform(sentences).toarray()
        labels = self.model.predict(sent_vecs)
        return [self.data.labels_names[labels[i]] for i in range(len(labels))]


