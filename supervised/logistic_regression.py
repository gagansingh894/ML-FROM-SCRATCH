import numpy as np


class LogisticRegression:
    

    def __init__(self, lr=0.001, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        self.n_samples, self.features = X.shape
        self.weights = np.zeros(self.features)
        self.bias = 0

        # GRADIENT DESCENT
        for _ in range(self.n_iters):
            self.linear_model = np.dot(X, self.weights) + self.bias
            self.y_predicted = self._sigmoid(self.linear_model.astype(np.float128))
        
            self.dw = (1 / self.n_samples) * np.dot(X.T, (self.y_predicted - y))
            self.db = (1 / self.n_samples) * np.sum(self.y_predicted - y)

            self.weights -= self.lr * self.dw
            self.bias -= self.lr * self.db

    def predict(self, X):
        self.linear_model = np.dot(X, self.weights) + self.bias
        self.predicted = self._sigmoid(self.linear_model)
        self.predicted_cls = [1 if i > 0.5 else 0 for i in self.predicted]
        return self.predicted_cls

    def _sigmoid(self, x):
        return  1 / (1 + np.exp(-x))
