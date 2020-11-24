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

    ## TODO: add evaluation

def test_qda():

    data = DGP(100,2)
    data.generate(usage='multiclassification')
    train_x, test_x, train_y, test_y = data.split()


    model = QDA()
    model.fit(train_x, train_y)
    pred = model.predict(test_x)

    ## TODO: add evaluation