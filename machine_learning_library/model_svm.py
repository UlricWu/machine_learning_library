from machine_learning_library.model_discriminant_analysis import Classification
import numpy as np


class SVM(Classification):
    def __init__(self, C=0.1, learning_rate=0.01, epochs=1000, threshold=0.0001, kernel=None, polynominal=2):
        super(SVM, self).__init__()

        self.C = C
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.threshold = threshold
        self.kernel = kernel
        self._polynominal = polynominal

    def _fit(self, features, labels):

        features = self._add_bias(features)
        features = self._get_kernel(features)
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
        features = self._get_kernel(features)

        return (features @ self._beta > 0).astype(int)

    def _check_converge(self, value):
        if np.sum(np.abs(value)) <= self.threshold:
            return True

        return False

    @staticmethod
    def _sign(labels):
        return np.where(labels > 0, 1, -1)

    def _get_kernel(self, x):
        if self.kernel is None:
            return x

        z = x
        if self.kernel == 'linear':
            return x @ (z.T)

        if self.kernel == 'polynominal':
            return np.power(1 + x @ z.T, self._polynominal)

        if self.kernel == 'rbf':
            ## use euclidean_dist_matrix K(x, y) = e^(-g||x - z||^2)
            norms_1 = (x ** 2).sum(axis=1)
            norms_2 = (z ** 2).sum(axis=1)
            distance = np.abs(norms_1.reshape(-1, 1) + norms_2 - 2 * np.dot(x, z.T))
            return np.exp(-self._polynominal * distance)
