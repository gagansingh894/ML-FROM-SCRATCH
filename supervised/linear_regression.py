import numpy as np 

class LinearRegression:

    def __init__(self, lr=0.001, n_iters=100):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None
    

    def fit(self, X, y):
        # Initialize the parameters
        self.n_samples, self.n_features = X.shape
        self.weights = np.zeros(self.n_features)
        self.bias = 0

        # Gradient Descent
        for _ in range(self.n_iters):
            self.y_predicted = np.dot(X, self.weights) + self.bias

            self.dw = (1/self.n_samples) * np.dot(X.T, (self.y_predicted - y))
            self.db = (1/self.n_samples) * np.sum(self.y_predicted - y)

            self.weights = self.lr * self.dw
            self.bias = self.lr * self.db

    def predict(self, X):
        self.predicted = np.dot(X, self.weights) + self.bias
        return self.predicted