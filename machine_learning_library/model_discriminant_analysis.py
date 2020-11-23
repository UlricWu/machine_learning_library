import numpy as np

class Classification:
    def __init__(self):
        pass

    def fit(self, features, labels):

        self._fit(features, labels)

    def predict(self, features):
        return self._predict(features)


    def _fit(self, features, labels):
        raise NotImplementedError

    def _predict(self, features):
        return NotImplementedError


class LDA(Classification):

    def __init__(self):
        super(LDA, self).__init__()

    def _fit(self, features, labels):

        variance = np.var(features, axis=0)
        identity = np.eye(features.shape[1])
        cov = identity * variance
        cov_inv = np.linalg.inv(cov)

        groups = np.unique(labels)
        indexs = {}
        means = {}
        for group in groups:

            index = np.nonzero(labels == group)[0]

            means[group] = features[index].mean(axis=0)
            indexs[group] = len(index)

        self._means = means
        self._groups = groups
        self._cov_inv = cov_inv
        self._indexs = indexs

    def _predict(self, features):
        likehood = np.zeros((features.shape[0], len(self._groups)))


        for group in self._groups:
            mean = self._means[group]
            prior_prob = self._indexs[group]
            score = features@self._cov_inv@mean.T +0.5*mean@self._cov_inv@mean.T + np.log(prior_prob)

            likehood[:,group] = score

        return likehood.argmax(axis=1).reshape(-1,1)


