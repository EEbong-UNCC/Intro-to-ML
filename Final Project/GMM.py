import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
import csv
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.mixture import GaussianMixture
from sklearn import metrics
from sklearn.cluster import KMeans
from scipy import stats
from linearregression import X_train_stand,X_test_stand,y_test,y_train,Y, X_train_norm, X_test_norm

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
        
        # Center data
        centered = cluster_data - means[k]
        
        # Compute covariance with regularization
        if covTYPE == "diag":
            cov = np.diag(np.var(centered, axis=0)) + 1e-6*np.eye(parsedData.shape[1])
        elif covTYPE == "full":
            cov = np.cov(centered.T) + 1e-6*np.eye(parsedData.shape[1])
        elif covTYPE == "tied":
            continue  # Handled separately
        elif covTYPE == "spherical":
            var = np.mean(np.var(centered, axis=0))
            cov = var * np.eye(parsedData.shape[1])
        else:
            raise ValueError(f"Unknown covariance type: {covTYPE}")
            
        covariances[k] = cov

    # Handle tied covariance
    if covTYPE == "tied":
        overall_centered = parsedData - np.mean(parsedData, axis=0)
        tied_cov = np.cov(overall_centered.T) + 1e-6*np.eye(parsedData.shape[1])
        covariances = np.array([tied_cov] * numclusters)

    # Compute precision matrices and their Cholesky decompositions
    precisions = np.zeros_like(covariances)
    precisions_cholesky = np.zeros_like(covariances)
    
    for k in range(numclusters):
        precisions[k] = np.linalg.inv(covariances[k])
        try:
            precisions_cholesky[k] = np.linalg.cholesky(precisions[k])
        except np.linalg.LinAlgError:
            # Add more regularization if Cholesky fails
            precisions[k] += 1e-6*np.eye(parsedData.shape[1])
            precisions_cholesky[k] = np.linalg.cholesky(precisions[k])

    # Create and initialize GMM
    gmm = GaussianMixture(
        n_components=numclusters,
        covariance_type=covTYPE,
        max_iter=0,  # Prevent further fitting
        init_params='random',  # Bypass scikit-learn's initialization
        random_state=42
    )
    
    # Manually set parameters
    gmm.weights_ = weights
    gmm.means_ = means
    gmm.covariances_ = covariances
    gmm.precisions_ = precisions
    gmm.precisions_cholesky_ = precisions_cholesky
    return gmm

if __name__ =='__main__':
    #run accurracy with GMM
    for x in range(2,5):
        gmmlist = kmmeanstrainGMM(X_train_norm,y_train,x)
        predictions = predict(gmmlist, X_test_norm)
        acc = accuracy(predictions, y_test)
        print(acc)

    

#GMM Most accurate with 2 clusters 
