from wu_machine_learning.machine_learning_library import data_generator
import numpy as np


def test_regression():
    observations, dimensions = 100000, 20
    data = data_generator.DGP(observations=observations, dimensions=dimensions)
    data.generate()
    train_x, test_x, train_y, test_y = data.split()
    assert train_x.shape == (70000, 20)
    assert test_x.shape == (30000, 20)
    assert train_y.shape == (70000, 1)
    assert test_y.shape == (30000, 1)


def test_binary():
    observations, dimensions = 100000, 20
    data = data_generator.DGP(observations=observations, dimensions=dimensions)
    data.generate(usage='binaryclassification')
    train_x, test_x, train_y, test_y = data.split()
    assert train_x.shape == (70000, 20)
    assert test_x.shape == (30000, 20)
    assert train_y.shape == (70000, 1)
    assert test_y.shape == (30000, 1)
    assert np.unique(test_y).tolist() == [0, 1]
    assert np.unique(train_y).tolist() == [0, 1]


def test_multi():
    observations, dimensions = 100000, 20
    data = data_generator.DGP(observations=observations, dimensions=dimensions)
    data.generate(usage='multiclassification')
    train_x, test_x, train_y, test_y = data.split()
    assert train_x.shape == (70000, 20)
    assert test_x.shape == (30000, 20)
    assert train_y.shape == (70000, 1)
    assert test_y.shape == (30000, 1)
    assert np.unique(test_y).tolist() == [0, 1, 2]
    assert np.unique(train_y).tolist() == [0, 1, 2]
