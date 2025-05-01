import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.mixture import GaussianMixture
from tslearn.clustering import TimeSeriesKMeans

#I should normalize data after i create the dataset not before dumb dumb dumb
# Import Data in such a way that the file name can be changed
data = []
person =[] 
instance_speeds = [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7]
for sub in range(1, 23):
    if sub == 4: 
        continue
    path_base = 'C:/Users/lolze/Documents/Spring 2025/Intro to ML/Github/Intro-to-ML/Final Project/Data/'
    for speed in instance_speeds:
        name = 'GP' + str(sub) + '_'+ str(speed) + '_force'
        path = path_base + name + '.csv'
        df = pd.read_csv(path)
        data.append(df.iloc[:,0:].values)
        person.append(sub)
data = np.array(data)
person = np.array(person)
X_train, X_test, y_train, y_test = train_test_split(data, person, test_size=0.20, random_state=2)
n_people = np.unique(person)

n_samples, n_time_steps, n_features = X_train.shape
n_clusters = 3
gmm_list = []

tskmeans = TimeSeriesKMeans(
    n_clusters=n_clusters,
    metric="euclidean"
)

for person_id in n_people: 
    person_data = X_train[y_train == person_id]
    tskmeans.fit(person_data)
    labels = tskmeans.labels_
    centers = tskmeans.cluster_centers_
    centers_flat = centers.reshape(n_clusters, -1)
    weights = []
    for k in range(n_clusters):
        frames = person_data[labels == k]
        weights.append(len(frames)/len(person_data))
        flat = frames.reshape(-1, frames.shape[1]*frames.shape[2])   
    gmm = GaussianMixture(
        n_components=n_clusters,
        covariance_type="full"
    )
    gmm.weights_ = weights
    gmm.means_ = centers_flat
    gmm_list.append(gmm)

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


predictions = predict(gmm_list, X_test)
acc = accuracy(predictions, y_test)
print(acc)


