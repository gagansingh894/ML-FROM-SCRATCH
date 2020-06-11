import numpy as np
from collections import Counter

def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1-x2)**2))

def manhattan_distance(x1, x2):
    return np.abs(np.sum(x2-x1))


class KNN:

    def __init__(self,k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        self.predicted_labels = [self._predict(x) for x in X]
        return np.array(self.predicted_labels)

    def _predict(self, x):
        # Compute distances
        self.distances = [euclidean_distance(x, x_train) for x_train in self.X_train]
        # Get K Nearest Neighbours
        self.k_indices = np.argsort(self.distances)[:self.k]
        # Get Labels
        self.k_nearest_labels = [self.y_train[i] for i in self.k_indices]
        # Majority Vote - Most Common Class Label
        self.most_common = Counter(self.k_nearest_labels).most_common(1)
        return self.most_common[0][0]