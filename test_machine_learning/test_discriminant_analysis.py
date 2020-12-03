from wu_machine_learning.machine_learning_library.data_generator import DGP
from wu_machine_learning.machine_learning_library.model_discriminant_analysis import LDA, QDA
from wu_machine_learning.machine_learning_library.model_evaluation import confusion_matirx

THRESHOLD = 1e-05


def test_lda():
    data = DGP()
    data.generate(usage='multiclassification')
    train_x, test_x, train_y, test_y = data.split()

    model = LDA()
    model.fit(train_x, train_y)
    pred = model.predict(test_x)

    assert confusion_matirx(test_y, pred).values.tolist() == [[9530, 0, 509],
                                                              [5022, 0, 5108],
                                                              [422, 0, 9409]]


def test_qda():
    data = DGP(100, 2)
    data.generate(usage='multiclassification')
    train_x, test_x, train_y, test_y = data.split()

    model = QDA()
    model.fit(train_x, train_y)
    pred = model.predict(test_x)

    assert confusion_matirx(test_y, pred).values.tolist() == [[2, 4, 1],
                                                              [0, 13, 1],
                                                              [0, 7, 2]]
