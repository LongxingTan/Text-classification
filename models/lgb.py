
from lightgbm import LGBMClassifier

class gbm:
    def __init__(self):
        pass

    def train(self,x_train,y_train):
        gbm=LGBMClassifier()
        gbm.fit(x_train,y_train)

    def evaluate(self,x_test,y_test):
        pass

    def predict(self,x_new):
        pass

    def plot(self):
        pass