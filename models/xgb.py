from xgboost import XGBClassifier
from sklearn import metrics
import pickle
import os

class xgb():
    def __init__(self):
        self.clf=None

    def train(self,x_train,y_train):
        self.x_train = x_train
        self.y_train = y_train
        self.clf = XGBClassifier(
            learning_rate=0.1,
            n_estimators=1000,
            max_depth=5,  # important to tune
            min_child_weight=1,  # important to tune
            gamma=0,  # important to tune
            subsample=0.8,
            colsample_bytree=0.8,
            objective='multi:softmax', #'binary:logistic'
            nthread=4,
            scale_pos_weight=1,
            reg_alpha=1e-05,  # L1
            reg_lambda=1,  # L2
            seed=27)
        self.clf.fit(self.x_train, self.y_train)
        pickle.dump(self.clf, open('xgbmodel.pkl', 'wb'))

    def evaluate(self,x_test,y_test):
        y_test_pre=self.clf.predict(x_test)
        y_test_pro=self.clf.predict_proba(x_test)[:, 1]
        print(y_test_pro,y_test_pre)
        #print("AUC Score : %f" % metrics.roc_auc_score(y_test, y_test_pro))
        print("Accuracy : %.4g" % metrics.accuracy_score(y_test, y_test_pre))

    def predict(self,x_new):
        if os.path.exists('xgbmodel.pkl'):
            xgbmodel=pickle.load(open('xgbmodel.pkl','rb'))
        else:
            xgbmodel =self.train()
        assert x_new.shape[1:]==self.x_train.shape[1:],'Invalid data input'
        y_new=xgbmodel.predict(x_new)
        return y_new

    def plot(self):
        pass
