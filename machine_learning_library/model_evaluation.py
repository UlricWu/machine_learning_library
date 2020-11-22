import numpy as np

def mean_square_error(real, predict):
    error = real - predict
    return np.mean(error@error.T)