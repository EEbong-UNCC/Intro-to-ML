#Eka Ebong 

import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
import csv
import seaborn as sns
from sklearn import preprocessing
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

file_path = 'C:/Users/lolze/Documents/Spring 2025/Intro to ML/Github/Intro-to-ML/Final Project/FullGaitFeatures.csv'

data = pd.read_csv(file_path)

#sepeate data into features and Y
X = data.iloc[:, 1:].values
Y = data.iloc[:, 0].values

#split data into train and test 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size = 0.20, random_state=2)

reg = LogisticRegression(penalty='l1',solver='liblinear')
reg.fit(X_train,y_train)
y_pred = reg.predict(X_test)

classes = reg.classes_
accuracy = metrics.accuracy_score(y_test,y_pred)
#print("Unprocessed Accuracy")
#print(accuracy)

# Display the confusion matrix
metrics.ConfusionMatrixDisplay.from_predictions(y_test,y_pred,labels=classes,
                                                       display_labels=[ 1,  2,  3,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]
                                                       , cmap="YlGnBu")
#plt.show()

#normalize data 

sc = preprocessing.MinMaxScaler()
X_train_norm = sc.fit_transform(X_train)
X_test_norm = sc.transform(X_test)
y_train_norm = sc.fit_transform(y_train.reshape(-1, 1)).flatten() 
y_test_norm = sc.transform(y_test.reshape(-1, 1)).flatten() 

reg2 = LogisticRegression()
reg2.fit(X_train_norm,y_train)
y_pred_2 = reg2.predict(X_test_norm)
accuracy_norm = metrics.accuracy_score(y_test,y_pred_2)

#print("Norm Accuracy")
#print(accuracy_norm)

metrics.ConfusionMatrixDisplay.from_predictions(y_test,y_pred_2,labels=classes
                                                       ,display_labels=[ 1,  2,  3,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]
                                                       ,cmap="YlGnBu")
#plt.show()


#standardize data
sc2 = preprocessing.StandardScaler()
X_train_stand = sc2.fit_transform(X_train)
X_test_stand = sc2.transform(X_test)
y_train_stand = sc2.fit_transform(y_train.reshape(-1, 1)).flatten() 
y_test_stand = sc2.transform(y_test.reshape(-1, 1)).flatten() 

reg3 = LogisticRegression(penalty='l1',solver='liblinear')
reg3.fit(X_train_stand,y_train)
y_pred_3 = reg3.predict(X_test_stand)
accuracy_stand = metrics.accuracy_score(y_test,y_pred_3)

#print("Stand Accuracy")
#print(accuracy_stand)
metrics.ConfusionMatrixDisplay.from_predictions(y_test,y_pred_3,labels=classes
                                                       ,display_labels=[ 1,  2,  3,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]
                                                       ,cmap="YlGnBu")
#plt.show()

#Naive Bayes with Standard
from sklearn.naive_bayes import GaussianNB
nb_classifier = GaussianNB()
nb_classifier.fit(X_train_stand, y_train)
nb_y_pred = nb_classifier.predict(X_test_stand)
acc_nb = metrics.accuracy_score(y_test, nb_y_pred)
nb_recall = metrics.recall_score(y_test,nb_y_pred, average=None)
nb_acc = metrics.accuracy_score(y_test,nb_y_pred)
nb_precision = metrics.precision_score(y_test,nb_y_pred, average=None)
nb_f1 = metrics.f1_score(y_test,nb_y_pred, average=None)
#print("Naive Bayes Accuracy")
#print(acc_nb)

#Build and SVM Classifier to classify the type of cancer 
from sklearn import svm
svm_classifier = svm.SVC(kernel='linear')
svm_classifier.fit(X_train_stand, y_train)
svm_y_pred = svm_classifier.predict(X_test_stand)

#Plot Classification accuracy, precision, recall and F1 score 
svm_acc = metrics.accuracy_score(y_test,svm_y_pred)
#print("SVM Accuracy: ", svm_acc)

from sklearn.ensemble import StackingClassifier
estimators = [('svm', svm.SVC(probability=True)), ('nb', GaussianNB())]
stack = StackingClassifier(estimators, final_estimator=LogisticRegression())
stack.fit(X_train_stand, y_train)
pred = stack.predict(X_test_stand)
acc2 = metrics.accuracy_score(y_test, pred)
#print("SVM+NB Accuracy: ", acc2)


'''
#PCA
pca = PCA(n_components=0.95)
X_pca = pca.fit_transform(X_train_stand)
X_pca_test = pca.transform(X_test_stand)
svm_classifier2 = svm.SVC(kernel='rbf')
svm_classifier2.fit(X_pca, y_train)
svm_y_pred2 = svm_classifier2.predict(X_pca_test)
svm_acc2 = metrics.accuracy_score(y_test,svm_y_pred2)
print("SVM Accuracy: ", svm_acc2)
'''

from sklearn.feature_selection import RFE 
from sklearn.pipeline import Pipeline


rfe = RFE(estimator=svm_classifier, n_features_to_select=10)
rfe.fit(X_train_stand,y_train)
#print(rfe.get_feature_names_out)

'''
from operator import itemgetter
column = ['Peak_Force_X1','Peak_Force_X2','Peak_Force_Y1','Peak_Force_Y2','Peak_Force_Z1','Peak_Force_Z2'
        ,'Mean_Force_X1','Mean_Force_X2','Mean_Force_Y1','Mean_Force_Y2','Mean_Force_Z1','Mean_Force_Z2'
        ,'Mean_FTI_Left', 'Mean_FTI_Right','Step_Duration','Average_Loading_Rate_Left','Average_Loading_Rate_Right'
        ,'Average_Unloading_Rate_Left','Average_Unloading_Rate_Right','Time_To_Peak_Force_Left','Time_To_Peak_Force_Right'
        ,'Double_Support_Time','Stance_Duration','Swing_Duration', 'FFT_Mean_Left', 'FFT_std_Left', 'FFT_Max_Left', 'FFT_Mean_Right'
        , 'FFT_std_Right', 'FFT_Max_Right', 'PSD_Peak_Freq_Left', 'PSD_Total_Power_Left', 'PSD_Band_Ratio_Left', 'PSD_Peak_Freq_Right'
        , 'PSD_Total_Power_Right', 'PSD_Band_Ratio_Right', 'Harmonic_Ratio_Left', 'Harmonic_Ratio_Right']
for x, y in (sorted(zip(rfe.ranking_ , column), key=itemgetter(0))):
    print(x, y)
'''

pred = rfe.predict(X_test_stand)
acc_rfe = metrics.accuracy_score(y_test,pred)
#print("RFE Accuracy", acc_rfe)

X_train_rfe = rfe.transform(X_train_stand)
X_test_rfe = rfe.transform(X_test_stand)

nb_2 = GaussianNB()
nb_2.fit(X_train_rfe, y_train)
nb2pred = nb_2.predict(X_test_rfe)
accnb2 = metrics.accuracy_score(y_test, nb2pred)
#print("RFE + NB Accuracy",x,": ", accnb2)

plt.close()
