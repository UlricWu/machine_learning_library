from machine_learning_library.model_discriminant_analysis import  Classification
import numpy as np


class LogsitcRegression(Classification):
    def __init__(self, learning_rate=0.01, threshold=1e-5, epochs=10000):
        super(LogsitcRegression, self).__init__()
        self.learning_rate = learning_rate
        self.threshold = threshold
        self.epochs = epochs
        self._beta= None


    def _fit(self, features, labels):
        features = _add_bias(features)
        beta = np.ones((features.shape[1], 1))

        for _ in range(self.epochs):
            x = features@beta
            p = sigmoid(x)
            jaco = jacobia(features, labels, p)
            hess = hession(features, p)

            update = np.linalg.inv(hess)@jaco

            if self._check_converge(update):
                self._beta = beta
                return
            beta += update

        self._beta=beta

    def _predict(self, features):
        features = _add_bias(features)
        prob =  sigmoid(features @ self._beta)
        return np.where(prob>0.5, 1, 0)

    def _check_converge(self, value):
        if np.sum(np.abs(value)) <= self.threshold:
            return True

        return False

def _add_bias(features):
    ones = np.ones((len(features), 1))
    return np.concatenate([ones, features], axis=1)

def jacobia(feature, label, p):
    return feature.T@(label - p)

def hession(feature, sigmoid):

    w = np.diagflat(sigmoid * (1 - sigmoid))
    return feature.T @ w @ feature

def sigmoid(x):
    return 1/(1 + np.exp(-x))