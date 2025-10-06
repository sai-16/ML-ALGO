import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


class LinearRegression1:
    def __init__(self):
        self.m = None
        self.c = None

    def fit(self, x_train, y_train):
        X = np.array(x_train)
        y = np.array(y_train)
        X_b = np.c_[np.ones((X.shape[0], 1)), X]

        # theta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
        # (xxT)-1*(xTy) can be also written in form of
        theta = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y
        self.c = theta[0]
        self.m = theta[1:]

    def predict(self, x_test):
        if self.m is None or self.c is None:
            raise ValueError('The Model is not fitted yet')
        X = np.array(x_test)
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        return X_b.dot(np.r_[self.c, self.m])

    def r2_score(self, x_test, y_test):
        y = np.array(y_test)
        y_pred = self.predict(x_test)
        tss = np.sum((y - np.mean(y)) ** 2)
        rss = np.sum((y - y_pred) ** 2)
        return 1 - (rss / tss)

    def mse(self, x_test, y_test):
        y_pred = self.predict(x_test)
        return np.mean((np.array(y_test) - np.array(y_pred)) ** 2)

    def mae(self, x_test, y_test):
        y_pred = self.predict(x_test)
        return np.mean(np.abs(np.array(y_test) - np.array(y_pred)))

    def coefficients(self):
        return self.m, self.c


class RidgeRegression:
    def __init__(self):
        self.m = None
        self.c = None
        self.alpha = None

    def fit(self, x_train, y_train):
        X = np.array(x_train)
        y = np.array(y_train)


class LassoRegression:
    def __init__(self):
        self.m = None
        self.c = None
        self.alpha = None
        self.alpha0 = None

    def fit(self, x_train, y_train):
        X = np.array(x_train)
        y = np.array(y_train)


class ElasticNet:
    def __init__(self):
        self.m = None
        self.c = None
        self.alpha = None
        self.alpha0 = None

    def fit(self, x_train, y_train):
        X = np.array(x_train)
        y = np.array(y_train)


if __name__ == "__main__":
    data = pd.read_csv(r"C:\Users\chiti\Machine Learning\ML ALGORITHMS\Regression\Theory\data\bmi.csv")
    data = pd.DataFrame(data)
    l = LabelEncoder()
    data["Gender"] = l.fit_transform(data['Gender'])
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=.30)
    m = LinearRegression1()
    m.fit(X_train, y_train)
    print(
        f'r2_score {m.r2_score(X_test, y_test)}  mean square error {m.mse(X_test, y_test)}  mean absolute error {m.mae(X_test, y_test)}')

