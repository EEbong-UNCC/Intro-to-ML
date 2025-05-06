import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
import csv
import seaborn as sns
from sklearn import preprocessing
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.feature_selection import RFE
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

file_path = 'C:/Users/lolze/Documents/Spring 2025/Intro to ML/Github/Intro-to-ML/Final Project/GaitFeatures.csv'

data = pd.read_csv(file_path)

#sepeate data into features and Y
X = data.iloc[:, 1:]
Y = data.iloc[:, 0].values

full_file_path = 'C:/Users/lolze/Documents/Spring 2025/Intro to ML/Github/Intro-to-ML/Final Project/FullGaitFeatures.csv'
datafull = pd.read_csv(full_file_path)

X_full = datafull.iloc[:,25:]
Y = datafull.iloc[:,0]

X_all = datafull.iloc[:, 1:]

picture_path = 'C:/Users/lolze/Documents/Spring 2025/Intro to ML/Github/Intro-to-ML/Final Project/Figure Folder/' 

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_all,Y, test_size = 0.20, random_state=42)
sc = preprocessing.StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
scaler = preprocessing.StandardScaler()
X_scaler = scaler.fit_transform(X_all)

column = ['Peak_Force_X1','Peak_Force_X2','Peak_Force_Y1','Peak_Force_Y2','Peak_Force_Z1','Peak_Force_Z2'
        ,'Mean_Force_X1','Mean_Force_X2','Mean_Force_Y1','Mean_Force_Y2','Mean_Force_Z1','Mean_Force_Z2'
        ,'Mean_FTI_Left', 'Mean_FTI_Right','Step_Duration','Average_Loading_Rate_Left','Average_Loading_Rate_Right'
        ,'Average_Unloading_Rate_Left','Average_Unloading_Rate_Right','Time_To_Peak_Force_Left','Time_To_Peak_Force_Right'
        ,'Double_Support_Time','Stance_Duration','Swing_Duration', 'FFT_Mean_Left', 'FFT_std_Left', 'FFT_Max_Left', 'FFT_Mean_Right'
        , 'FFT_std_Right', 'FFT_Max_Right', 'PSD_Peak_Freq_Left', 'PSD_Total_Power_Left', 'PSD_Band_Ratio_Left', 'PSD_Peak_Freq_Right'
        , 'PSD_Total_Power_Right', 'PSD_Band_Ratio_Right', 'Harmonic_Ratio_Left', 'Harmonic_Ratio_Right']

#split data into train and test 
def timefeaturecorrelation():
    plt.figure(figsize=(11, 8))
    correlation_matrix = X.corr(method='pearson')
    mask = np.triu(np.ones_like(correlation_matrix))
    ax = sns.heatmap(correlation_matrix, mask=mask, cmap='PiYG', vmin=-1, vmax=1)
    plt.title('Time Features Correlation')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=10)

    plt.tight_layout()
    name = 'TimeFeatCorr.png'
    plt.savefig(picture_path+name, dpi=300)
    plt.show()

def fullfeaturecorrelation():
    plt.figure(figsize=(11, 8))
    correlation_matrix = X_full.corr(method='pearson')
    mask = np.triu(np.ones_like(correlation_matrix))
    ax = sns.heatmap(correlation_matrix, annot=True, mask=mask, cmap='PiYG', vmin=-1, vmax=1)
    plt.title('Frquency Features Correlation')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=10)

    plt.tight_layout()
    name = 'FreqFeatCorr.png'
    plt.savefig(picture_path+name, dpi=300)
    plt.show()

def allfeatcorrelation():
    plt.figure(figsize=(11, 8))
    correlation_matrix = X_all.corr(method='pearson')
    mask = np.triu(np.ones_like(correlation_matrix))
    ax = sns.heatmap(correlation_matrix, mask=mask, cmap='PiYG', vmin=-1, vmax=1)
    plt.title('All Features Correlation')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=10)
    plt.tight_layout()
    name = 'AllFeatCorr.png'
    plt.savefig(picture_path+name, dpi=300)
    plt.show()

def rfe(x):
    rfe = RFE(estimator=SVC(kernel='linear'), n_features_to_select=x)
    rfe.fit(X_train, y_train)
    return rfe

def featureimportance():
    recFE = rfe(10)
    importance_df = pd.DataFrame({
        'Feature': column,
        'Ranking': rfe.ranking_
    }).sort_values('Ranking')

    plt.figure(figsize=(11,8))
    sns.heatmap(importance_df.set_index('Feature'), cmap='Blues', cbar=False)
    plt.title('RFE Feature Rankings')
    plt.show()

def bestModelPred():
    recfe = rfe(12)
    X_train_rfe = recfe.transform(X_train)
    X_test_rfe = recfe.transform(X_test)

    NB= SVC(kernel='linear')
    NB.fit(X_train_rfe, y_train)
    NB_pred = NB.predict(X_test_rfe)
    return NB_pred

def confusionBest():
    nb = bestModelPred()
    ax = metrics.ConfusionMatrixDisplay.from_predictions(y_test,nb,labels=[ 1,  2,  3,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]
                                                       ,display_labels=[ 1,  2,  3,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]
                                                       ,cmap="Blues")
    fig = ax.ax_.get_figure()
    fig.set_figheight(10)
    fig.set_figwidth(10)
    plt.tight_layout()
    plt.title("RFE Confusion Matrix")
    name = 'confusionmatrix.png'
    plt.savefig(picture_path+name, dpi=300)
    plt.show()

def cross_validation(reg_model, X, Y, cv):
    scores = cross_val_score(
        reg_model, X, Y, 
        scoring='accuracy', cv=cv
    )
    rmse_scores = scores.mean()
    return rmse_scores

def evaluatenumFeat(x):
    acc = []
    for num in range(1,x):
        features = rfe(num)
        xrf = features.transform(X_train)
        xrft = features.transform(X_test)
        nb = GaussianNB()
        nb.fit(xrf, y_train)
        pred = nb.predict(xrft)
        accuracy = metrics.accuracy_score(y_test, pred)
        acc.append(accuracy)
    return acc

def evaluatenumFeatRFE(x):
    acc = []
    for num in range(1,x):
        features = rfe(num)
        xrf = features.transform(X_train)
        xrft = features.transform(X_test)
        nb = SVC(kernel='linear')
        nb.fit(xrf, y_train)
        pred = nb.predict(xrft)
        accuracy = metrics.accuracy_score(y_test, pred)
        acc.append(accuracy)
    return acc

def evaluatenumFeatValidated(x):
    n_splits = 5
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    acc = []
    for num in range(1,x):
        svm_classifier = SVC(kernel='linear')
        model = RFE(estimator=svm_classifier, n_features_to_select=num)
        model.fit(X_scaler, Y)
        new_X = model.transform(X_scaler)
        new_svm = GaussianNB()
        accuracy = cross_validation(new_svm, new_X, Y, cv =kf)
        acc.append(accuracy)
    return acc

def evaluatenumFeatValidatedRFE(x):
    n_splits = 5
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    acc = []
    for num in range(1,x):
        svm_classifier = SVC(kernel='linear')
        model = RFE(estimator=svm_classifier, n_features_to_select=num)
        model.fit(X_scaler, Y)
        new_X = model.transform(X_scaler)
        new_svm = SVC(kernel='linear')
        accuracy = cross_validation(new_svm, new_X, Y, cv =kf)
        acc.append(accuracy)
    return acc

def accuracyovertime(accuracies, name, title):
    num = len(accuracies)+1
    indexmax = np.argmax(accuracies)+1
    max = np.max(accuracies)
    plt.figure(figsize=(12, 8))
    plt.plot(
        range(1,num), 
        accuracies, 
        color='#FF6B6B',  # Coral color (replace with any hex/name)
        marker='o', 
        linestyle='-', 
        linewidth=2,
        markersize=8,
        label='Model Accuracy'
    )

    # Highlight peak at 10 features
    plt.axvline(x=indexmax, color='#4ECDC4', linestyle='--', alpha=0.7, label=f'Optimal ({indexmax:.1f} Features)')
    plt.annotate(
        f'Peak Accuracy: {max:.3f}', 
        xy=(indexmax, max), 
        xytext=(indexmax+2, max - 0.07),
        arrowprops=dict(facecolor='black', shrink=0.06),
        fontsize=12
    )

    # Labels and styling
    plt.xlabel("Number of Features Selected by RFE", fontsize=12)
    plt.ylabel("Accuracy", fontsize=12)
    plt.title(title, fontsize=14, pad=20)
    plt.xticks(range(1,num))
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.legend(framealpha=1)
    plt.tight_layout()

    # Save and show
    name = name
    plt.savefig(picture_path+name, dpi=300, bbox_inches='tight')
    plt.show()

#accuracyovertime(evaluatenumFeatRFE(41),"accuracy_vs_featuresRFE.png", "Accuracy vs. Number of Selected Features with RFE Model")
#accuracyovertime(evaluatenumFeatValidated(41),"accuracy_vs_featuresval.png", "Mean K-Fold Validation Accuracy vs. Number of Selected Features")
#accuracyovertime(evaluatenumFeatValidatedRFE(41),"accuracy_vs_featuresvalRFE.png", "Mean K-Fold Validation Accuracy vs. Number of Selected Features RFE Model")
confusionBest()