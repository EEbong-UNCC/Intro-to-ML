#Eka Ebong 

import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv('C:/Users/lolze/Documents/Spring 2025/Intro to ML/Github/Intro-to-ML/Assignment 2/Housing.csv')

#sepeate data into features and Y
X = data.iloc[:, 1:13].values
Y = data.iloc[:, 0].values

#split data into train and test 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size = 0.20, random_state=0)

#Problem 1

#TODO develop linear regression with gradient decent algorithm 
def cost_function(theta, X, y): 
    m = len(y)
    prediction = X.dot(theta)
    error = np.subtract(prediction,y)
    sqrerror = np.square(error)
    cost = 1/(2*m) * np.sum(sqrerror)
    return cost 

def gradient_descent(xtrain,xtest,ytrain,ytest, learning_rate, iterations, theta, intercept=False):
    myX = xtrain
    myXtest = xtest
    if intercept == True: 
        #add zero row 
        X0 = np.ones(np.shape(xtrain)[0])
        X1 = np.asarray(xtrain)
        myX = np.c_[X0,X1]
        X0 = np.ones(np.shape(xtest)[0])
        X1 = np.asarray(xtest)
        myXtest = np.c_[X0,X1]
    m = len(ytrain)
    cost_history_train = []
    cost_history_test = []
    for i in range(iterations):
        predictions = myX.dot(theta)
        errors = np.subtract(predictions, ytrain) 
        gradients = (learning_rate/m)*myX.transpose().dot(errors)
        theta = np.subtract(theta,gradients)
        cost_history_train.append(cost_function(theta,myX,ytrain))
        cost_history_test.append(cost_function(theta,myXtest,ytest))
    return theta, cost_history_train,cost_history_test

#TODO evaluate based on area, bedrooms, bathrooms, stories, parking 
all_features = ['Area', 'bedrooms', 'bathrooms', 'stories', 'mainroad', 'guestroom', 'basement', 'hotwaterheat', 'aircondition', 'parking', 'prefarea', 'furnishingstatus']
oneafeatures = [ 'bedrooms', 'bathrooms', 'stories', 'parking']
def featurelist(currentfeatures):
    featureindex = []
    all_features = ['Area', 'bedrooms', 'bathrooms', 'stories', 'mainroad', 'guestroom', 'basement', 'hotwaterheat', 'aircondition', 'parking', 'prefarea', 'furnishingstatus']
    for x in currentfeatures:
        featureindex.append(all_features.index(x))
    return featureindex
X1a = X_train[:, featurelist(oneafeatures)].astype('float64')
X1a_test = X_test[:, featurelist(oneafeatures)].astype('float64')
theta_1a = np.zeros(len(oneafeatures) + 1)
theta_1a, cost_history1a, cost_history1a_test = gradient_descent(X1a,X1a_test, y_train.astype('float64'), y_test.astype('float64'),0.01,50,theta_1a,intercept=True)

plt.plot(np.linspace(0,50,50), cost_history1a, label='Train')
plt.plot(np.linspace(0,50,50), cost_history1a_test, label='Test')
plt.legend()
plt.show()
print(cost_history1a)

#TODOExplore values for learning rate
#TODO Evaluate based on Area, bedrooms, bathrooms, stories, main road, duestroom, basement, hot water heating, air conditioning, parking, prefarea?
onebfeatures = ['Area', 'bedrooms', 'bathrooms', 'stories', 'mainroad', 'guestroom', 'basement', 'hotwaterheat', 'aircondition', 'parking', 'prefarea']
onebselectfeatures = ['Area']
#all numerical values, consider leaving out parking this time? 
X1b = X_train[:, featurelist(onebselectfeatures)].astype('float64')
X1b_test = X_test[:, featurelist(onebselectfeatures)].astype('float64')
theta_1b = np.zeros(len(onebselectfeatures)+ 1)
theta_1b, cost_history1b, cost_history1b_test = gradient_descent(X1b, X1b_test, y_train.astype('float64'), y_test.astype('float64'), 0.01, 50, theta_1b, intercept=True)

plt.plot(np.linspace(0,50,50), cost_history1b, label='Train1b')
plt.plot(np.linspace(0,50,50), cost_history1b_test, label='Test1b')
plt.legend()
plt.show()
print(cost_history1b)
#TODO identify best paramteres for linear regression 
#TODO plot the traning and validation losses (MSE) 
#TODO explore learning rates 

#Problem 2

#TODO normalize data 
#TODO standardize data
#TODO Normalized: evaluate based on area, bedrooms, stories, parking
#TODO Standardized: evaluate based on area, bedrooms, stories, parking
#TODO plot training and validation losses
#TODO compare scaling approaches an which achieves the best training 
#TODO Normalized: Evaluate based on Area, bedrooms, bathrooms, stories, main road, duestroom, basement, hot water heating, air conditioning, parking, prefarea?
#TODO Standardized: Evaluate based on Area, bedrooms, bathrooms, stories, main road, duestroom, basement, hot water heating, air conditioning, parking, prefarea?
#TODO plot training and validation losses

#Problem 3 
#TODO Repeat 2a with parameter penalties (make sure to modify gradient descent logic on training set)) 
#TODO PLot results 
#TODO Repeat 2b with parameter penalties
#Plot Results