import numpy as np

def token(text):
    return text.split()

def representation(words):
    unique_words = list(set(words))

    dictionary = {}

    for value in words:
        matrix = [0] * len(unique_words)
        matrix[unique_words.index(value)] = 1

        dictionary[value] = matrix

    return dictionary

def validate(x, y, dictionary):
    for i in x:
        print([k for k,v in dictionary.items() if v in i])

    print('-------------')
    for i in y:
        print([k for k,v in dictionary.items() if v == i])

def main(text='the quick brown fox jumps over the lazy dog',
         window=2,
         size=5
         ):

    words = token(text)

    dictionary = representation(words)

    x, y= onehot_encode(dictionary, window, words)

    # validate(x,y, dictionary)
    #
    new_x = np.array([np.mean(i, axis=0) for i in x])

    new_y = np.array(y)

    weight_1, weight_2 = np.random.random((len(dictionary), size)), np.ones((size, len(dictionary)))

    learning_rate = 0.001
    # for _ in range(1):
    for _ in range(100):
        H = new_x@weight_1
        u = (H@weight_2)
        pred = softmax(u)

        error = pred - new_y

        weight_2 -= learning_rate * H.T@error
        weight_1 -= learning_rate * new_x.T@(error@weight_2.T)

    print(weight_1)


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)




def onehot_encode(dictionary, window, words):
    search_window = [i for i in range(-window, window + 1) if i != 0]
    x, y = [], []
    for i in range(len(words)):
        y.append(dictionary[words[i]])

        x.append([dictionary[words[pos + i]] for pos in search_window
                  if -1 < (pos + i) < len(words)])

    return x, y

if __name__ == '__main__':
    main()