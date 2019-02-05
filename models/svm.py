from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import roc_curve, auc,f1_score,precision_recall_curve,precision_score
from sklearn.externals import joblib
from sklearn.model_selection import GridSearchCV,ShuffleSplit,cross_val_score
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from models_tf_archives._utils import *
import os

class svm():
    def __init__(self,x_train,y_train):
        self.x_train=x_train
        self.y_train=y_train
        self.svmmodel=SVC(C=1.0, kernel="linear", probability=True, decision_function_shape='ovo')

    def train(self):
        self.svmmodel.fit(self.x_train, self.y_train)
        joblib.dump(self.svmmodel,'svm.pkl')
        return self.svmmodel



    def evaluate(self,x_test,y_test):
        if os.path.exists('svm.pkl'):
            svmmodel=joblib.load('svm.pkl')
        else:
            svmmodel =self.train()
        y_test_predict = svmmodel.predict(x_test)
        print('precision score',precision_score(y_test,y_test_predict,average='macro'))
        print('f1 score:',f1_score(y_test,y_test_predict,average='macro'))
        y_test_prob = svmmodel.predict_proba(x_test)
        y_test_onehot = label_binarize(y_test, np.arange(len(set(y_train))))  # n_classes
        fpr, tpr, thresholds = roc_curve(y_test_onehot.ravel(), y_test_prob.ravel())
        ax = plt.subplot(2, 1, 1)
        ax.step(fpr, tpr, color='b', alpha=0.2, where='post')
        ax.fill_between(fpr, tpr, step='post', alpha=0.2, color='b')
        ax.set_title('ROC curve')

        precision, recall, th = precision_recall_curve(y_test_onehot.ravel(), y_test_prob.ravel())
        ax = plt.subplot(2, 1, 2)
        ax.plot(recall, precision)
        ax.fill_between(recall, precision, step='post', alpha=0.2, color='b')
        ax.set_ylim([0.0, 1.0])
        ax.set_title('Precision recall curve')
        plt.show()

    def predict(self,x_new):
        if os.path.exists('svm.pkl'):
            svmmodel=joblib.load('svm.pkl')
        else:
            svmmodel =self.train()
        assert x_new.shape[1:]==self.x_train.shape[1:],'Invalid data input'
        y_new=svmmodel.predict(x_new)
        return y_new

    def _cross_validation(self):
        cv=ShuffleSplit(n_splits=5,test_size=0.2, random_state=20)
        scores=cross_val_score(self.svmmodel,self.x_train,self.y_train,cv=cv,scoring='f1_macro')
        print(scores)

    def _grid_search(self):
        c_range = np.logspace(-2, 10, 13)
        parameters={'C':c_range}
        grid_cls=GridSearchCV(estimator=self.svmmodel,param_grid=parameters,n_jobs=1,verbose=2)
        grid_cls.fit(self.x_train,self.y_train)
        self.svmmodel.set_params(C=grid_cls.best_params_['C'])