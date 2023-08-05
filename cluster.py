
import numpy as np
import random
from sklearn.metrics import pairwise_distances


class Cluster:
    def __init__(self, X) -> None:
        self.X = X

    def init_centroids(self, K, dimension):
        centroids = np.zeros((K,dimension))

        for i in range(K):
            cen = np.zeros(dimension)
            for d in range(dimension):
                min_range = min(self.X.iloc[:,d])
                max_range = max(self.X.iloc[:,d])
                cen[d] = random.uniform(min_range, max_range)
                centroids[i] = cen
        return centroids
    
    def assign_labels(self, centroids, similarity):
        distances = pairwise_distances(self.X, centroids, metric = similarity)
        
        labels = [np.argmin(d) for d in distances]
        #print(labels)
        return labels
    
    def update_centroids(self, labels, centroids):
        K = len(centroids)
        clusters = [[] for i in range(K)]
        for i, label in enumerate(labels):
            clusters[label].append(self.X.iloc[i])
        
        dimension = self.X.shape[1]
        centroids = np.zeros((K,dimension))
        for i in range(K):
            cluster = np.array(clusters[i])
            if len(cluster) == 0:
                continue
            cen = np.zeros(dimension)
            for d in range(dimension):
                cen[d] = np.mean(cluster[:,d])
            centroids[i] = cen
        return centroids

    #centroids = plus or random
    def Kmeans(self, K=3, maxiter=100, similarity = 'cosine'):
        dimension = self.X.shape[1]
        ## initialization to do
        cur_centroids = self.init_centroids(K, dimension)
        #print(cur_centroids)
        prev_centroids = np.zeros((K,dimension))

        for ii in range(maxiter):
            labels = self.assign_labels(cur_centroids, similarity)
            prev_centroids = cur_centroids
            cur_centroids = self.update_centroids(labels, cur_centroids)


            comparison = cur_centroids == prev_centroids
            if type(comparison)==bool:
                if comparison:
                    break
                else:
                    equal_arrays = comparison.all()
                if equal_arrays:
                    break

        return labels, cur_centroids

    