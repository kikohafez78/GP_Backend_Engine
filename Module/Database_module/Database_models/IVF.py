import numpy as np
from scipy.cluster.vq import kmeans2
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial import distance
import cvxopt
import copy
import pandas as pd

def sort_vectors_by_cosine_similarity(vectors, reference_vector):
    # Calculate cosine similarities
    cos_similarities = cosine_similarity(reference_vector, vectors)
    # Sort indices by cosine similarity
    sorted_indices = np.argsort(cos_similarities)

    return sorted_indices


def sort_list(list1, list2):
    zipped_pairs = zip(list2, list1)

    z = [x for _, x in sorted(zipped_pairs)]

    return z


# clusters = ceil(len(vectors)/sqrt(len(vectors)) * 3
class IVFile(object):
    def __init__(self, partitions: int, vectors: np.ndarray):
        self.partitions = partitions
        self.vectors = vectors

    def clustering(self):
        kmeans = KMeans(n_clusters=self.partitions)
        assignments = kmeans.fit_predict(self.vectors)
        centroids = kmeans.cluster_centers_
        # (centroids, assignments) = kmeans2(self.vectors, self.partitions)
        self.data = (centroids, assignments)
        index = [[] for _ in range(self.partitions)]
        for n, k in enumerate(assignments):
            index[k].append(self.vectors[n])
        centroid_assignment = {}
        x = 0
        for k in index:
            byte_file = np.asarray(k)
            centroid_assignment[str(centroids[x])] = f"./data/file{x}.npy"
            file = open(f"./data/file{x}.npy", "a")
            np.save(f"./data/file{x}.npy", byte_file)
            file.close()
            x += 1
            k = None
        self.assigments = centroid_assignment
        self.vectors = None  # <===
        return self.assigments

    def get_closest_centroids(self, vector: np.ndarray, K: int):
        centroids = self.data[0]
        index = sort_vectors_by_cosine_similarity(centroids, vector)
        centroids = centroids[index]
        return centroids[0][len(centroids) - K - 1 :]

    def get_cluster_data(self, centroid: np.ndarray, vector: np.ndarray, K: int):
        data = np.load(self.assigments[str(centroid)])
        index = sort_vectors_by_cosine_similarity(data, vector)
        data = data[index]
        return data[0][len(data) - K - 1 :]

    def cluster_data(self, centroids: np.ndarray):
        return [np.load(self.assigments[str(centroid)]) for centroid in centroids]

    def get_closest_k_neighbors(self, vector: np.ndarray, K: int):
        centroids = self.get_closest_centroids(vector, K)
        closest = []
        for centroid in centroids:
            closest.append(self.get_cluster_data(centroid, vector, K))
        closest = np.asarray(closest)
        closest = closest.reshape((closest.shape[0] * closest.shape[1], closest.shape[2]))
        indices = sort_vectors_by_cosine_similarity(closest, vector)
        return [closest[i] for i in indices[0][len(closest) - K - 1 :]]

    def get_K_closest_neighbors_given_centroids(self, centroids: np.ndarray, vector: np.ndarray, K: int):
        closest = []
        for centroid in centroids:
            closest.append(self.get_cluster_data(centroid, vector, K))
        closest = np.asarray(closest)
        closest = closest.reshape((closest.shape[0] * closest.shape[1], closest.shape[2]))
        indices = sort_vectors_by_cosine_similarity(closest, vector)
        return [closest[i] for i in indices[0][len(closest) - K - 1 :]]

    def get_K_closest_neigbors_inside_centroid_space(self, centroids: np.ndarray, vector: np.ndarray, K: int):
        centroids = self.get_closest_centroids(vector, 1)
        closest = []
        for centroid in centroids:
            closest.append(self.get_cluster_data(centroid, vector, K))
        closest = np.asarray(closest)
        closest = closest.reshape((closest.shape[0] * closest.shape[1], closest.shape[2]))
        indices = sort_vectors_by_cosine_similarity(closest, vector)
        return [closest[i] for i in indices[0][len(closest) - K - 1 :]]


class IVFTree(object):
    def __init__(self, partitions: int, vectors: np.ndarray):
        self.partitions = partitions
        self.nlevels = np.floor(np.log2(partitions))
        self.vectors = vectors

    def create_tree(self):
        self.graph: list[IVFile] = []
        self.assignments: list[list[dict]] = []
        vectors = self.vectors
        for i in range(self.nlevels, -1, 1):
            iv = IVFile(i**2, vectors)
            cluster_assignment = iv.clustering()
            self.assignments.append(cluster_assignment)
            vectors = [np.fromstring(cluster) for cluster in cluster_assignment.keys()]
            self.graph.append(iv)

    def find(self, vector: np.ndarray, K: int):
        level0 = self.graph[0]
        clusters = vector
        for iv in reversed(self.graph):
            if iv != level0:
                clusters = iv.get_closest_k_neighbors(clusters, 1)
            else:
                clusters = iv.get_K_closest_neighbors_given_centroids(clusters, vector, K)
        return clusters


class IVFile_optimized(object):
    def __init__(self, vectors: np.ndarray):
        self.partitions = np.ceil(len(vectors) / np.sqrt(len(vectors))) * 3
        self.vectors = vectors

    def clustering(self):
        (centroids, assignments) = kmeans2(self.vectors, self.partitions)
        self.data = (centroids, assignments)
        index = [[] for _ in range(self.partitions)]
        for n, k in enumerate(assignments):
            index[k].append(self.vectors[n])
        centroid_assignment = {}
        x = 0
        for k in index:
            byte_file = np.asarray(k)
            centroid_assignment[str(centroids[x])] = f"./classes/class{x}.npy"
            file = open(f"./classes/class{x}.npy", "a")
            np.save(f"./classes/class{x}.npy", byte_file)
            file.close()
            x += 1
            k = None
        self.assigments = centroid_assignment
        self.vectors = None  # <===
        return self.assigments

    def get_closest_centroids(self, vector: np.ndarray, K: int):
        centroids = self.data[0]
        index = sort_vectors_by_cosine_similarity(centroids, vector)
        centroids = centroids[index]
        return centroids[0][len(centroids) - K - 1 :]

    def get_cluster_data(self, centroid: np.ndarray, vector: np.ndarray, K: int):
        data = np.load(self.assigments[str(centroid)])
        index = sort_vectors_by_cosine_similarity(data, vector)
        data = data[index]
        return data[0][len(data) - K - 1 :]

    def cluster_data(self, centroids: np.ndarray):
        return [np.load(self.assigments[str(centroid)]) for centroid in centroids]

    def get_closest_k_neighbors(self, vector: np.ndarray, K: int):
        centroids = self.get_closest_centroids(vector, K)
        closest = []
        for centroid in centroids:
            closest.append(self.get_cluster_data(centroid, vector, K))
        closest = np.asarray(closest)
        closest = closest.reshape((closest.shape[0] * closest.shape[1], closest.shape[2]))
        indices = sort_vectors_by_cosine_similarity(closest, vector)
        return [closest[i] for i in indices[0][len(closest) - K - 1 :]]

    def get_K_closest_neighbors_given_centroids(self, centroids: np.ndarray, vector: np.ndarray, K: int):
        closest = []
        for centroid in centroids:
            closest.append(self.get_cluster_data(centroid, vector, K))
        closest = np.asarray(closest)
        closest = closest.reshape((closest.shape[0] * closest.shape[1], closest.shape[2]))
        indices = sort_vectors_by_cosine_similarity(closest, vector)
        return [closest[i] for i in indices[0][len(closest) - K - 1 :]]

    def get_K_closest_neigbors_inside_centroid_space(self, centroids: np.ndarray, vector: np.ndarray, K: int):
        centroids = self.get_closest_centroids(vector, 1)
        closest = []
        for centroid in centroids:
            closest.append(self.get_cluster_data(centroid, vector, K))
        closest = np.asarray(closest)
        closest = closest.reshape((closest.shape[0] * closest.shape[1], closest.shape[2]))
        indices = sort_vectors_by_cosine_similarity(closest, vector)
        return [closest[i] for i in indices[0][len(closest) - K - 1 :]]


class SVM:
    linear = lambda x, y , c=0: x @ y.T
    polynomial = lambda x, y, Q=5: (1 + x @ y.T)**Q
    rbf = lambda x, z, γ=10: np.exp(-γ*distance.cdist(x, z,'sqeuclidean'))
    kernel_funs = {'linear': linear, 'polynomial': polynomial, 'rbf': rbf}
    def __init__(self, kernel: str = "rbf", C: int = 1, k = 6, multiclass: bool = True):
        self.kernel_fun = kernel
        self.kernel = self.kernel_funs[kernel]
        self.C = C
        self.k = k
        self.X, Y = None, None
        self.alpha = None
        self.multiclass = multiclass
        self.clfs = []
        
    def fit(self, X, y, eval_train: bool = False):
        if len(np.unique(y)) > 2:
            self.multiclass = True
            return self.multi_fit(X, y, eval_train)
        if set(np.unique(y)) == {0, 1}: y[y == 0] = -1
        self.y = y.reshape(-1, 1).astype(np.double)
        self.X = X
        N = X.shape[0]  
        self.K = self.kernel(X, X, self.k)
        P = cvxopt.matrix(self.y @ self.y.T * self.K)
        q = cvxopt.matrix(-np.ones((N, 1)))
        A = cvxopt.matrix(self.y.T)
        b = cvxopt.matrix(np.zeros(1))
        G = cvxopt.matrix(np.vstack((-np.identity(N),
                                    np.identity(N))))
        h = cvxopt.matrix(np.vstack((np.zeros((N,1)),
                                    np.ones((N,1)) * self.C)))
        cvxopt.solvers.options['show_progress'] = False
        sol = cvxopt.solvers.qp(P, q, G, h, A, b)
        self.alpha = np.array(sol["x"])         
        self.is_sv = ((self.alpha-1e-3 > 0)&(self.alpha <= self.C)).squeeze()
        self.margin_sv = np.argmax((0 < self.alpha-1e-3)&(self.alpha < self.C-1e-3))
        if eval_train:  
            print(f"Finished training with accuracy{self.evaluate(X, y)}")

    def multi_fit(self, X, y, eval_train=False):
        self.k = len(np.unique(y))
        for i in range(self.k):
            Xs, Ys = X, copy.copy(y)
            Ys[Ys!=i], Ys[Ys==i] = -1, +1
            clf = SVM(kernel=self.kernel_fun, C=self.C, k=self.k)
            clf.fit(Xs, Ys)
            self.clfs.append(clf)
            
        if eval_train:  
            print(f"Finished training with accuracy {self.evaluate(X, y)}")

    def multi_predict(self, X):
        N = X.shape[0]
        preds = np.zeros((N, self.k))
        for i, clf in enumerate(self.clfs):
            _, preds[:, i] = clf.predict(X)
        return np.argmax(preds, axis=1), np.max(preds, axis=1)

   
    def predict(self, X_t):
        if self.multiclass: return self.multi_predict(X_t)
        xₛ, yₛ = self.X[self.margin_sv, np.newaxis], self.y[self.margin_sv]
        alpha, y, X= self.alpha[self.is_sv], self.y[self.is_sv], self.X[self.is_sv]
        b = yₛ - np.sum(alpha * y * self.kernel(X, xₛ, self.k), axis=0)
        score = np.sum(alpha * y * self.kernel(X, X_t, self.k), axis=0) + b
        return np.sign(score).astype(int), score
    
    def evaluate(self, X,y):  
        outputs, _ = self.predict(X)
        accuracy = np.sum(outputs == y) / len(y)
        return round(accuracy, 2)
    
    
