#Assignment 3 

#Import Breast Cancer Dataset 

import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import load_breast_cancer
breast = load_breast_cancer()
X = breast.data
Y = breast.target
#Scale data properly and Standardize
from sklearn import preprocessing

stan = preprocessing.StandardScaler()
X_stand = stan.fit_transform(X)
input = pd.DataFrame(X_stand)


#Problem 1
#Split 30 inputs 80% train 20% test
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(input,Y, test_size = 0.2, random_state=0)

# Build loogistic regression model to classify types of cancer
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train, Y_train)
Y_pred = classifier.predict(X_test)

#Report results including accuracy precision and recall 
from sklearn import metrics
recall = metrics.recall_score(Y_test,Y_pred)
acc = metrics.accuracy_score(Y_test,Y_pred)
precision = metrics.precision_score(Y_test,Y_pred)

#Plot the confusion matrix representing pinary classifier 
metrics.ConfusionMatrixDisplay.from_predictions(Y_test,Y_pred)
plt.title('Confusion Matrix for Logistic Regression')
plt.show()
print("Accuracy: ", acc)
print("Recall: ", recall)
print("Precision: ", precision)

#Part 2

#Add a weight penalty
#weight_class = LogisticRegression(random_state=0, penalty='l1', solver='liblinear')
weight_class = LogisticRegression(random_state=0, penalty='l2')
#With L2 penalty, the results are the same

#Repeat training 
weight_class.fit(X_train,Y_train)
wc_y_pred = weight_class.predict(X_test)

#Report results including accuracy precision and recall 
wc_recall = metrics.recall_score(Y_test,wc_y_pred)
wc_acc = metrics.accuracy_score(Y_test,wc_y_pred)
wc_precision = metrics.precision_score(Y_test,wc_y_pred)
print("Weight Penalty Accuracy: ", wc_acc)
print("Weight Penalty Recall: ", wc_recall)
print("Weight Penalty Precision: ", wc_precision)
#Plot the confusion matrix representing pinary classifier 
metrics.ConfusionMatrixDisplay.from_predictions(Y_test,wc_y_pred)
plt.title('Confusion Matrix for Logistic Regression with Weight Penalties')
plt.show()

#Problem 2 
#Build naive Bayesian Model 
from sklearn.naive_bayes import GaussianNB
nb_classifier = GaussianNB()
nb_classifier.fit(X_train,Y_train)
nb_y_pred = nb_classifier.predict(X_test)

#Plot Classification accuracy, precision, recall and F1 score 
nb_recall = metrics.recall_score(Y_test,nb_y_pred)
nb_acc = metrics.accuracy_score(Y_test,nb_y_pred)
nb_precision = metrics.precision_score(Y_test,nb_y_pred)
nb_f1 = metrics.f1_score(Y_test,nb_y_pred)
print("Naive Bayes Accuracy: ", nb_acc)
print("Naive Bayes Recall: ", nb_recall)
print("Naive Bayes Precision: ", nb_precision)
print("Naive Bayes F1: ", nb_f1)

# Function to add value labels on top of bars
def add_labels(x, y):
    for i in range(len(x)):
        plt.text(i, y[i], round(y[i],2)) # Placing text slightly above the bar

plt.bar(['Recall', 'Accuracy','Precision','F1'],[nb_recall,nb_acc,nb_precision,nb_f1],color=['tab:red', 'tab:blue', 'tab:green','tab:orange'])
plt.ylabel('Percentage')
add_labels(['Recall', 'Accuracy','Precision','F1'],[nb_recall,nb_acc,nb_precision,nb_f1])
plt.title('Recall, Accuracy, Precision and F1 score for Naive Bayes Classifier')
plt.show()
metrics.ConfusionMatrixDisplay.from_predictions(Y_test,nb_y_pred)
plt.title('Confusion Matrix for Logistic Regression with Naive Bayes')
plt.show()

#Explain and elaborate on results and compare to problem 1
'''
The results for the Naive Bayes model show that the model is less accurate, has less recall precision and F1 score than the previous 2 models.
We can conclude from this that there is some type of weak link between some of the features as Naive Bayes assumes that the each feature is conditionally independent 
of every other feature. Looking at the features of the dataset, things such as radius, perimeter and area are likely to be dependent on one another.
'''

#Problem 3 
#Build and SVM Classifier to classify the type of cancer 
from sklearn import svm
svm_classifier = svm.SVC(kernel='linear')
svm_classifier.fit(X_train, Y_train)
svm_y_pred = svm_classifier.predict(X_test)

#Plot Classification accuracy, precision, recall and F1 score 
svm_recall = metrics.recall_score(Y_test,svm_y_pred)
svm_acc = metrics.accuracy_score(Y_test,svm_y_pred)
svm_precision = metrics.precision_score(Y_test,svm_y_pred)
svm_f1 = metrics.f1_score(Y_test,svm_y_pred)
print("SVM Accuracy: ", svm_acc)
print("SVM Recall: ", svm_recall)
print("SVM Precision: ", svm_precision)
print("SVM F1: ", svm_f1)

plt.bar(['Recall', 'Accuracy','Precision','F1'],[svm_recall,svm_acc,svm_precision,svm_f1],color=['tab:red', 'tab:blue', 'tab:green','tab:orange'])
add_labels(['Recall', 'Accuracy','Precision','F1'],[svm_recall,svm_acc,svm_precision,svm_f1])
plt.ylabel('Percentage')
plt.title('Recall, Accuracy, Precision and F1 score for SVM')
plt.show()
metrics.ConfusionMatrixDisplay.from_predictions(Y_test,svm_y_pred)
plt.title('Confusion Matrix for Logistic Regression with SVM')
plt.show()

#Explain and elaborate on results and compare to problem 1
'''
The support vector method is the best model so far. This makes sense as it is an improvement of our logistic regression, so it makes sense that it would be a better result. Additionally 
we know that there are some dependencies between features so the SVM out performs the Naive Bayes as well. 
'''

#Problem 4
#TODO Use PCA feature extraction for training
from sklearn.decomposition import PCA
results = []

#TODO Perform N number of independent training 
for k in range(1,31):
    pca = PCA(n_components=k)
    Xpca_train = pca.fit_transform(X_train)
    Xpca_test = pca.transform(X_test)
    pca_classifier = LogisticRegression(random_state=0)
    pca_classifier.fit(Xpca_train, Y_train)
    pca_y_pred = pca_classifier.predict(Xpca_test)

    results.append({
        'k':k,
        'Accuracy': metrics.accuracy_score(Y_test,pca_y_pred),
        'Recall': metrics.recall_score(Y_test,pca_y_pred),
        'Precision': metrics.precision_score(Y_test,pca_y_pred),
        'F1': metrics.f1_score(Y_test,pca_y_pred)
    })
#identify optimum K 
#Optimum k = 7 according to graph.
#Plot Classification accuracy, precision, recall and F1 score 
plt.plot([r['k'] for r in results], [r['Accuracy'] for r in results], label='Accuracy')
plt.plot([r['k'] for r in results], [r['Precision'] for r in results], label='Precision')
plt.plot([r['k'] for r in results], [r['Recall'] for r in results], label='Recall')
plt.plot([r['k'] for r in results], [r['F1'] for r in results], label='F1 Score')
plt.xlabel('Number of Principal Components (K)')
plt.ylabel('Score')
plt.legend()
plt.title('Performance Metrics vs. Number of PCA Components')
plt.show()
print(results[6])
#Explain and elaborate on results and compare to problem 1 and Problem 2
'''
Accuracy:  0.9649122807017544
Recall:  0.9701492537313433
Precision:  0.9701492537313433
Weight Penalty Accuracy:  0.956140350877193
Weight Penalty Recall:  0.9701492537313433
Weight Penalty Precision:  0.9558823529411765
Naive Bayes Accuracy:  0.9035087719298246
Naive Bayes Recall:  0.9104477611940298
Naive Bayes Precision:  0.9242424242424242
Naive Bayes F1:  0.9172932330827067
SVM Accuracy:  0.9736842105263158
SVM Recall:  0.9850746268656716
SVM Precision:  0.9705882352941176
SVM F1:  0.9777777777777777
{'k': 7, 'Accuracy': 0.9649122807017544, 'Recall': 0.9701492537313433, 'Precision': 0.9701492537313433, 'F1': 0.9701492537313433}

Compared to Problem 1, 2, and 3, the PCA reduced set works quite the same as the original logistic regression but not quite as good as the support vector machine. 
However, by reducing the number of variables, the model is faster and still maintains around the same level of precision, accuracy and most importantly recall. In the case of 
tumor diagnosis, this is the most important factor as we want to catch as many malignant cases as possible. 
'''