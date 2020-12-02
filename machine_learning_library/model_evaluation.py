import numpy as np
import pandas as pd


def mean_square_error(real, predict):
    error = real - predict
    return np.mean(error @ error.T)


def confusion_matirx(test, pred):
    groups = np.unique(test)

    index_true = {i: [] for i in groups}

    ## index of each class of true labels
    for index, label in enumerate(test):

        index_true[label[0]].append(index)

    counts = []
    ## groupby classes of true labels, count the numbers for each class in predict
    for group in groups:
        count = {i: 0 for i in groups}
        for label in pred[index_true[group]]:
            count[label[0]] += 1
        counts.append(count)

    matirx = pd.DataFrame(counts)

    matirx.columns = ['predict ' + str(group) for group in groups]

    return matirx
