from machine_learning_library.model_discriminant_analysis import  Classification
import numpy as np


class LogsitcRegression(Classification):
    def __init__(self, learning_rate=0.01, threshold=1e-5, epochs=10000):
        super(LogsitcRegression, self).__init__()
        self.learning_rate = learning_rate
        self.threshold = threshold
        self.epochs = epochs
        self.beta= None


    def _fit(self, features, labels, method='MLE'):
        beta = np.ones((features.shape[1], 1))

        for _ in range(self.epochs):
            x = features@beta
            p = sigmoid(x)
            jaco = jacobia(features, labels, p)
            hess = hession(features, p)

            update = np.linalg.inv(jaco)@hess

            if self._check_converge(update):
                self.beta = beta
                return
            beta += update

        self.beta=beta

    def _predict(self, features):
        return sigmoid(features@self.beta)


    def _check_converge(self, value):
        if value <= self.threshold:
            return True

        return False



def jacobia(feature, label, p):
    return feature.T@(label - p)
def hession(feature, beta):
    weight = np.diag(beta*(1-beta))
    return feature.T@weight@weight

def sigmoid(x):
    return 1/(1 + np.exp(-x))