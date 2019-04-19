# ******************************************************************************
# classification.py
#
# Date      Name       Description
# ========  =========  ========================================================
# 3/27/19      Initial version,
# ******************************************************************************
#import visualization tools
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mat
from mpl_toolkits.mplot3d import Axes3D


#import scikit-learn library
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_predict
from imblearn.over_sampling import SMOTE
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier

class Classification():
    def __init__(self):
        pass

    '''Function to print result of the classifier'''
    def print_result(self, y_test, y_pred):
        target_names = ['Low Growth', 'High Growth']
        print(metrics.classification_report(y_test, y_pred, target_names=target_names))
        print("Confusion Matrix: \n", metrics.confusion_matrix(y_test, y_pred))

    def run_logistic_regression(self, X, y):
        X = preprocessing.scale(X)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                            random_state=42)  # 70% training and 30% test
        log = linear_model.LogisticRegression(solver='liblinear')
        log.fit(X_train, y_train)

        #check if the model is correct in train data
        print("\n\nTraining")
        y_pred = log.predict(X_train)
        self.print_result(y_train, y_pred)

        print("\n\nTesting")
        # now predict on test data
        y_pred = log.predict(X_test)
        self.print_result(y_test, y_pred)

    def run_logistic_cross_val(self, X, y):
        X = preprocessing.scale(X)
        log = linear_model.LogisticRegression(solver='liblinear')
        y_pred = cross_val_predict(log, X, y, cv=10)
        self.print_result(y, y_pred)

    def correlation_heatmap(self, final_dataset):
        correlations = final_dataset.corr()
        fig, ax = plt.subplots(figsize=(10, 10))
        sns.heatmap(correlations, vmax=1.0, center=0, fmt='.2f',
                    square=True, linewidths=.5, annot=True, cbar_kws={"shrink": .70})
        plt.savefig('figures/corelation_heatmap.png')

    def run_logistic_regression_with_resampling(self, X, y):
        sm = SMOTE(random_state=12)
        X = preprocessing.scale(X)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                            random_state=42)  # 70% training and 30% test

        # oversample the training set
        x_train_res, y_train_res = sm.fit_sample(X_train, y_train)
        x_test_res, y_test_res = sm.fit_sample(X_test, y_test)

        log = linear_model.LogisticRegression(solver='liblinear')
        log.fit(x_train_res, y_train_res)

        # check if the model is correct in train data
        print("\n\nTraining")
        y_pred = log.predict(x_train_res)
        self.print_result(y_train_res, y_pred)
        print("\n\nTesting")
        # now predict on test data
        y_pred = log.predict(X_test)
        self.print_result(y_test, y_pred)

    def run_decision_tree(self, X, y):
        #X = preprocessing.scale(X)
        decision_tree_classifier = DecisionTreeClassifier()
        y_pred = cross_val_predict(decision_tree_classifier, X, y, cv=10)
        self.print_result(y, y_pred)

    def run_random_forest(self, X, y):
        #X = preprocessing.scale(X)
        random_forest_classifier = RandomForestClassifier(n_estimators=10, criterion="entropy")
        y_pred_rf = cross_val_predict(random_forest_classifier, X, y, cv=10)
        self.print_result(y, y_pred_rf)
        return y_pred_rf

    def run_svm(self, X, y):
        X = preprocessing.scale(X)
        svm_classifier = svm.SVC(kernel="linear")
        y_pred = cross_val_predict(svm_classifier, X, y, cv=10)
        self.print_result(y, y_pred)
