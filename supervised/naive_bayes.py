import numpy as np

class NaiveBayes:

    def fit(self, X, y):
       self.n_samples, self.n_features = X.shape
       self._classes = np.unique(y) # [1,2,3]
       self.n_classes = len(self._classes) # 3
       
       #init mean, varaince and priors for each class
       self._mean = np.zeros((self.n_classes, self.n_features), dtype=np.float64)
       self._var = np.zeros((self.n_classes, self.n_features), dtype=np.float64)
       self._priors = np.zeros(self.n_classes, dtype=np.float64)
       print(self._mean)
       print('*********')
       print(self._var)
       print('*********')
       print(self._priors)
       print('*********')

       for c in self._classes:
           print(c)
           self.X_c = X[y==c]
           print(self.X_c)
           self._mean[c,:] = self.X_c.mean(axis=0)
           self._var[c,:] = self.X_c.var(axis=0)
           self._priors[c] = self.X_c.shape[0] / float(self.n_samples)
           print(self._mean)
           print('*********')
           print(self._var)
           print('*********')
           print(self._priors)
           print('*********')
    
    def predict(self, X):
        y_pred = [self._predict(x) for x in X]
        return y_pred
    
    def _predict(self, x):
        self.posteriors = []

        for idx, c in enumerate(self._classes):
            self.prior = np.log(self._priors[idx])
            self.class_conditional = np.sum(np.log(self._pdf(idx,x)))
            self.posterior = self.prior + self.class_conditional
            self.posteriors.append(self.posterior)
        
        return self._classes[np.argmax(self.posteriors)]

    def _pdf(self, class_idx, x):
        self.mean = self._mean[class_idx]
        self.var = self._var[class_idx]
        self.num = np.exp(-(x-self.mean)**2 / (2 * self.var))
        self.denom = np.sqrt(2*np.pi * self.var)
        return self.num/ self.denom