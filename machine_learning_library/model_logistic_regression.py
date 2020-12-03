from machine_learning_library.model_discriminant_analysis import Classification
import numpy as np


class LogsitcRegression(Classification):
    def __init__(self, learning_rate=0.01, threshold=1e-5, epochs=10000, method='MLE'):
        super(LogsitcRegression, self).__init__()
        self.learning_rate = learning_rate
        self.threshold = threshold
        self.epochs = epochs
        self._beta = None
        self.method = method

    def _fit(self, features, labels):
        features = self._add_bias(features)
        beta = np.ones((features.shape[1], 1))

        for _ in range(self.epochs):
            x = features @ beta
            p = self._sigmoid(x)
            update = self._mle(features, labels, p) if self.method == 'MLE' else \
                self.learning_rate * self._jacobia(features, labels, p)

            if self._check_converge(update):
                self._beta = beta
                return
            beta += update

        self._beta = beta

    def _mle(self, features, labels, p):
        jaco = self._jacobia(features, labels, p)
        hess = self._hession(features, p)
        update = np.linalg.inv(hess) @ jaco
        return update

    def _predict(self, features):
        features = self._add_bias(features)
        prob = self._sigmoid(features @ self._beta)
        return np.where(prob > 0.5, 1, 0)

    def _check_converge(self, value):
        if np.sum(np.abs(value)) <= self.threshold:
            return True

        return False

    @staticmethod
    def _add_bias(features):
        ones = np.ones((len(features), 1))
        return np.concatenate([ones, features], axis=1)

    @staticmethod
    def _jacobia(feature, label, p):
        return feature.T @ (label - p)

    @staticmethod
    def _hession(feature, sigmoid):

        w = np.diagflat(sigmoid * (1 - sigmoid))
        return feature.T @ w @ feature

    @staticmethod
    def _sigmoid(x):
        return 1 / (1 + np.exp(-x))
