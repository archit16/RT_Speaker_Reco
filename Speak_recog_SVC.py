from __future__ import division
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from datetime import datetime
import matplotlib
import math
from sklearn import ensemble
from joblib import dump, load

train_file = pd.read_csv('speak_reco_DSP_new.csv')
test_file = pd.read_csv('speak_reco_DSP_test.csv')
features = ['feat_1','feat_2','feat_3','feat_4','feat_5','feat_6','feat_7','feat_8','feat_9','feat_10','feat_11','feat_12','feat_13']

X_train = train_file[features].values
Y_train = train_file['Speaker'].values
X_test = test_file[features].values
Y_test = test_file['Speaker'].values

#Standardising the values.
std_scaler_x = preprocessing.StandardScaler().fit(X_train)
X_train_std = std_scaler_x.transform(X_train)
X_test_std = std_scaler_x.transform(X_test)
# X_train_std = X_train
# X_test_std = X_test
Y_train_std = Y_train
Y_test_std = Y_test


# Prediction Algorithms -


# SVR algorithm
SVM_clf = svm.SVC(kernel='rbf', degree = 3)
SVM_clf.fit(X_train_std, Y_train_std)
dump(SVM_clf, 'speakereco_model.joblib') 
SVM_accuracy = SVM_clf.score(X_test_std, Y_test_std)
SVM_accuracytrain = SVM_clf.score(X_train_std, Y_train_std)
y_SVM_test_prediction = SVM_clf.predict(X_test_std)

print (y_SVM_test_prediction)
print ('SVM Accuracy = '+str(SVM_accuracy))
print ('SVM Train Accuracy = '+str(SVM_accuracytrain))
