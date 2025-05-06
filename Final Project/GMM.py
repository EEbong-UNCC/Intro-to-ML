import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.mixture import GaussianMixture
from sklearn import preprocessing
from sklearn.cluster import KMeans
from scipy import stats

file_path = 'C:/Users/lolze/Documents/Spring 2025/Intro to ML/Github/Intro-to-ML/Final Project/GaitFeatures.csv'

data = pd.read_csv(file_path)

#sepeate data into features and Y
X = data.iloc[:, 1:].values
Y = data.iloc[:, 0].values

#split data into train and test 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size = 0.20, random_state=2)

#standardize data
sc2 = preprocessing.StandardScaler()
X_train_stand = sc2.fit_transform(X_train)
X_test_stand = sc2.transform(X_test)
y_train_stand = sc2.fit_transform(y_train.reshape(-1, 1)).flatten() 
y_test_stand = sc2.transform(y_test.reshape(-1, 1)).flatten() 

file_path2 = 'C:/Users/lolze/Documents/Spring 2025/Intro to ML/Github/Intro-to-ML/Final Project/FullGaitFeatures.csv'

data2 = pd.read_csv(file_path2)

#sepeate data into features and Y
X2 = data.iloc[:, 1:].values
Y2 = data.iloc[:, 0].values

#split data into train and test 
from sklearn.model_selection import train_test_split
X_train2, X_test2, y_train2, y_test2 = train_test_split(X2,Y2, test_size = 0.20, random_state=2)

#standardize data
sc22 = preprocessing.StandardScaler()
X_train_stand2 = sc22.fit_transform(X_train2)
X_test_stand2 = sc22.transform(X_test2)
y_train_stand2 = sc22.fit_transform(y_train2.reshape(-1, 1)).flatten() 
y_test_stand2 = sc22.transform(y_test2.reshape(-1, 1)).flatten() 


def trainGMM(x_train, y_train, num_clusters):
    n_people = np.unique(Y)
    gmm_models = []
    for person_id in n_people:
        person_data = x_train[y_train==person_id]
        gmm = GaussianMixture(n_components=num_clusters)
        gmm.fit(person_data)
        gmm_models.append(gmm)
    return gmm_models

n_components_range = range(1, 10)
bic_scores = []
n_people = np.unique(Y)

def predict(gmmlist, testData):
    predictions = np.zeros(len(testData))
    for index, sample in enumerate(testData):
        likelihoods = []
        for model in gmmlist:
            score = model.score(sample.reshape(1,-1))
            likelihoods.append(score)
        likelihoods = np.asarray(likelihoods)
        predictions[index] = n_people[np.argmax(likelihoods)]
    return predictions

def accuracy(predictions, testlabels):
    accuracy = (predictions == testlabels).sum() / len(testlabels)
    return accuracy

# Predict on test data
#y_pred = np.array([predict_person(x, gmm_models) for x in X_test])
#accuracy = accuracy_score(y_test, y_pred)
#print(f"Accuracy: {accuracy:.2f}")
def clusternum():
    n_components_range = range(1, 10)
    bic_scores = []
    for n in n_components_range:
        gmm = GaussianMixture(n_components=n, random_state=42)
        gmm.fit(X_train_stand)
        bic_scores.append(gmm.bic(X_train_stand))

    best_n = n_components_range[np.argmin(bic_scores)]
    print(f"Best n_components: {best_n}")

def kmmeanstrainGMM(x_train, y_train, num_clusters):
    n_people = np.unique(Y)
    gmm_models = []
    for person_id in n_people:
        person_data = x_train[y_train==person_id]
        gmm = kmeansGMM(person_data, num_clusters, "full")
        gmm.fit(person_data)
        gmm_models.append(gmm)
    return gmm_models

def kmeansGMM(parsedData, numclusters, covTYPE):
    kmeans = KMeans(n_clusters=numclusters, n_init=10).fit(parsedData)
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_

    weights = np.zeros(numclusters)
    means = np.zeros((numclusters, parsedData.shape[1]))
    covariances = np.zeros((numclusters, parsedData.shape[1], parsedData.shape[1]))
    
    # Compute cluster statistics
    for k in range(numclusters):
        cluster_data = parsedData[labels == k]
        weights[k] = len(cluster_data) / len(parsedData)
        means[k] = np.mean(cluster_data, axis=0)
    # Create and initialize GMM
    gmm = GaussianMixture(
        n_components=numclusters,
        covariance_type=covTYPE,
        weights_init=weights,
        means_init= means,
    )
    
    # Manually set parameters
    return gmm

if __name__ =='__main__':
    #run accurracy with GMM
    print("----- GMM Model Accuracy Time Domain Features ------")
    gmmlist = trainGMM(X_train_stand,y_train,2)
    predictions = predict(gmmlist, X_test_stand)
    acc = accuracy(predictions, y_test)
    print("----- GMM Model Accuracy EM ------")
    print(acc)

    kgmmlist =kmmeanstrainGMM(X_train_stand, y_train,2)
    kpreditions = predict(kgmmlist, X_test_stand)
    kacc = accuracy(predictions, y_test)
    print("----- GMM Model Accuracy K-Means ------")
    print(kacc)

    print("----- GMM Model Accuracy All Features ------")
    gmmlist2 = trainGMM(X_train_stand2,y_train2,2)
    predictions2 = predict(gmmlist2, X_test_stand2)
    acc2 = accuracy(predictions2, y_test2)
    print("----- GMM Model Accuracy EM ------")
    print(acc2)

    kgmmlist2 =kmmeanstrainGMM(X_train_stand2, y_train2,2)
    kpreditions2 = predict(kgmmlist2, X_test_stand2)
    kacc2 = accuracy(predictions2, y_test2)
    print("----- GMM Model Accuracy K-Means ------")
    print(kacc2)
     
