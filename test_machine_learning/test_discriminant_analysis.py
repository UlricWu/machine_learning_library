from wu_machine_learning.machine_learning_library.data_generator import DGP
from wu_machine_learning.machine_learning_library.model_discriminant_analysis import LDA
from wu_machine_learning.machine_learning_library.model_evaluation import mean_square_error

THRESHOLD = 1e-05
def test_lda():

    data = DGP()
    data.generate(usage='multiclassification')
    train_x, test_x, train_y, test_y = data.split()


    model = LDA()
    model.fit(train_x, train_y)
    pred = model.predict(test_x)

    assert (mean_square_error(test_y, pred) - 7.511111e-05) <=THRESHOLD
