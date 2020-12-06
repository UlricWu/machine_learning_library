from machine_learning_library.model_discriminant_analysis import Classification
import numpy as np


class SVM(Classification):
    def __init__(self, C=0.1, learning_rate=0.01, epochs=1000, threshold=0.0001):
        super(SVM, self).__init__()

        self.C = C
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.threshold = threshold

    def _fit(self, features, labels):

        features = self._add_bias(features)
        beta = np.zeros((features.shape[1], 1))

        labels = self._sign(labels)
        for epoch in range(self.epochs * len(features)):
            update = beta if labels.T @ features @ beta >= 1 else \
                beta - self.C * features.T @ labels

            if self._check_converge(update):
                self._beta = beta
                return

            beta -= self.learning_rate * update

        self._beta = beta

    def _predict(self, features):
        features = self._add_bias(features)

        return (features @ self._beta > 0).astype(int)

    def _check_converge(self, value):
        if np.sum(np.abs(value)) <= self.threshold:
            return True

        return False

    @staticmethod
    def _sign(labels):
        return np.where(labels > 0, 1, -1)
