from wu_machine_learning.machine_learning_library import model_linear_regression
from wu_machine_learning.machine_learning_library import data_generator
from wu_machine_learning.machine_learning_library import model_evaluation

THRESHOLD = 1e-5

def load_data(observations=100000, dimensions=20):
    data = data_generator.DGP(observations=observations, dimensions=dimensions)
    data.generate()
    train_x, test_x, train_y, test_y = data.split()

    return train_x, test_x, train_y, test_y

def test_least_square():
    train_x, test_x, train_y, test_y = load_data()

    model = model_linear_regression.LeastSquare()
    model.fit(train_x, train_y)
    pred = model.predict(test_x)
    assert (model_evaluation.mean_square_error(test_y, pred)- 1.730196e-06) <= THRESHOLD

def test_ridge():
    train_x, test_x, train_y, test_y = load_data()

    model = model_linear_regression.Ridge()
    model.fit(train_x, train_y)
    pred = model.predict(test_x)

    assert (model_evaluation.mean_square_error(test_y, pred)- 1.730175e-06) <= THRESHOLD

def test_gradient_descent():
    train_x, test_x, train_y, test_y = load_data()

    model = model_linear_regression.GradientDescent()
    model.fit(train_x, train_y)
    pred = model.predict(test_x)

    assert (model_evaluation.mean_square_error(test_y, pred) - 1.731382e-06) <= THRESHOLD


