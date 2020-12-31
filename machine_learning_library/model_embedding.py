import numpy as np

class Embedding:
    def __init__(self, window=2, size=5, epoch=100):
        self.window = window
        self.size = size
        self.epoch = epoch

    def fit(self, text):
        words = self._token(text)

        self._representation(words)

        weight_1, weight_2 = self._nerual_network(words)

        self._weight1 = weight_1
        self._weight2 = weight_2

    def predict(self):
        return self._weight1


    def _nerual_network(self, words):
        x, y = self._onehot_encode(words)
        new_x = np.array([np.mean(i, axis=0) for i in x])
        new_y = np.array(y)
        weight_1, weight_2 = np.random.random((len(self._dictionary), self.size)), np.ones((self.size, len(self._dictionary)))
        learning_rate = 0.001
        for _ in range(self.epoch):
            H = new_x @ weight_1
            u = (H @ weight_2)
            pred = self._softmax(u)

            error = pred - new_y

            weight_2 -= learning_rate * H.T @ error
            weight_1 -= learning_rate * new_x.T @ (error @ weight_2.T)
        return weight_1, weight_2

    @staticmethod
    def _token(text):
        return text.split()

    def _representation(self, words):
        unique_words = list(set(words))

        dictionary = {}

        for value in words:
            matrix = [0] * len(unique_words)
            matrix[unique_words.index(value)] = 1

            dictionary[value] = matrix

        self._dictionary = dictionary

    @staticmethod
    def _softmax(x):
        """Compute softmax values for each sets of scores in x."""
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)

    def _onehot_encode(self, words):
        search_window = [i for i in range(-self.window, self.window + 1) if i != 0]
        x, y = [], []
        for i in range(len(words)):
            y.append(self._dictionary[words[i]])

            x.append([self._dictionary[words[pos + i]] for pos in search_window
                      if -1 < (pos + i) < len(words)])

        return x, y



if __name__ == '__main__':
    text='the quick brown fox jumps over the lazy dog'
    model = Embedding()
    model.fit(text)
    print(model.predict())