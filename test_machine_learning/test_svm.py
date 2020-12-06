from wu_machine_learning.machine_learning_library.model_svm import SVM
from wu_machine_learning.machine_learning_library.data_generator import DGP
from machine_learning_library.model_evaluation import confusion_matirx


def test_svm():
    observations, dimensions = 1000, 5
    data = DGP(observations, dimensions)
    data.generate(usage='binaryclassification')
    train_x, test_x, train_y, test_y = data.split()

    model = SVM()

    model.fit(train_x, train_y)

    pred = model.predict(test_x)

    assert confusion_matirx(test_y, pred).values.tolist() == [[103, 51],
                                                              [32, 114]]
