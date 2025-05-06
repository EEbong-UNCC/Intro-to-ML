#Eka Ebong 

import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
import csv
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, cross_validate
from sklearn.feature_selection import RFE
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score


file_path = 'C:/Users/lolze/Documents/Spring 2025/Intro to ML/Github/Intro-to-ML/Final Project/FullGaitFeatures.csv'

data = pd.read_csv(file_path)

#sepeate data into features and Y
X = data.iloc[:, 1:].values
Y = data.iloc[:, 0].values

scaler = StandardScaler()
X_scaler = scaler.fit_transform(X)

cnt = 0 
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
p_score = make_scorer(metrics.precision_score,average='macro')

from sklearn.model_selection import cross_val_score

def cross_validation(reg_model, X, Y, cv):
    scores = cross_val_score(
        reg_model, X, Y, 
        scoring='accuracy', cv=cv
    )
    rmse_scores = scores
    print("Scores:", rmse_scores)
    print("Mean:", rmse_scores.mean())
    print("StandardDeviation:", rmse_scores.std())

scorers = {
    'accuracy': 'accuracy',
    'precision': make_scorer(precision_score, average='macro'),
    'recall': make_scorer(recall_score, average='macro'),
    'f1': make_scorer(f1_score, average='macro')
}

def fullcross_validation(reg_model, X, Y, cv):
    # Use cross_validate to compute multiple metrics
    scores = cross_validate(
        reg_model, X, Y, 
        scoring=scorers, cv=cv, return_train_score=False
    )
    
    # Print results for each metric
    for metric in ['accuracy', 'precision', 'recall', 'f1']:
        print(f"{metric.capitalize()}:")
        print(f"  Scores: {scores[f'test_{metric}']}")
        print(f"  Mean: {np.mean(scores[f'test_{metric}']):.4f}")
        print(f"  Std: {np.std(scores[f'test_{metric}']):.4f}")
    print("----------------------")


print("----- SVM Model Cross Validation ------")

svm_classifier = svm.SVC(kernel="linear")
#cross_validation(svm_classifier, X_scaler, Y, kf)
fullcross_validation(svm_classifier, X_scaler, Y, kf)

print("----- NB Model Cross Validation ------")
nb_classifier = GaussianNB()
#cross_validation(nb_classifier, X_scaler, Y, kf)
fullcross_validation(nb_classifier, X_scaler, Y, kf)

rfe = RFE(estimator=svm_classifier, n_features_to_select=12)
rfe.fit(X_scaler,Y)
X_rfe = rfe.transform(X_scaler)

print("----- SVM + RFE Model Cross Validation ------")

svm_classifier = svm.SVC(kernel="linear")
#cross_validation(svm_classifier, X_rfe, Y, kf)
fullcross_validation(svm_classifier, X_rfe, Y, kf)

print("----- RFE SVM and NB Model Cross Validation ------")
nb_class2 = GaussianNB()
#cross_validation(nb_class2, X_rfe, Y, kf)
fullcross_validation(nb_class2, X_rfe, Y, kf)