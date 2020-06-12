import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt
import os
import sys
sys.path.append(os.getcwd())

X, y = datasets.make_regression(n_samples=100, n_features=2, noise=20, random_state=4)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

# fig = plt.figure(figsize=(8,6))
# plt.scatter(X[:, 0], y, color="b", marker="o", s=30)
# plt.show()

# print(y_train)
# print(y_train.shape)

from supervised import linear_regression

lr = linear_regression.LinearRegression()
lr.fit(X_train, y_train)
pred = lr.predict(X_test)

def mse(y_true, y_pred):
    return  np.mean(np.square((y_true - y_pred)))

print("Mean Squared Error: {}".format(mse(y_test, pred)))