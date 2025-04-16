#Eka Ebong 

import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
import csv
import seaborn as sns

file_path = 'C:/Users/lolze/Documents/Spring 2025/Intro to ML/Github/Intro-to-ML/Final Project/GaitFeatures.csv'

data = pd.read_csv(file_path)

#sepeate data into features and Y
X = data.iloc[:, 1:27].values
Y = data.iloc[:, 0].values

#split data into train and test 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size = 0.20, random_state=2)

from sklearn.linear_model import LogisticRegression
from sklearn import metrics
#classes = reg.classes_

#standardize data
from sklearn import preprocessing
sc2 = preprocessing.StandardScaler()
X_train = sc2.fit_transform(X_train)
X_test = sc2.transform(X_test)

from sklearn import svm
clf = svm.SVC(kernel='linear', C=1).fit(X_train, y_train)
clf.score(X_test, y_test)

#metrics.ConfusionMatrixDisplay.from_predictions(y_test,y_pred_3,labels=classes
                                                       ,display_labels=[ 1,  2,  3,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]
                                                       ,cmap="YlGnBu")
#plt.show()
#TODO Implement cross validation 
#TODO implement PCA
#TODO Implement feature selection 
#TODO Implement K mean 
#TODO Naive Bayes
