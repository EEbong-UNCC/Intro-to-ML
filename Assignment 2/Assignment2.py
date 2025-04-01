#Eka Ebong 

import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
import csv

file_path = 'C:/Users/lolze/Documents/Spring 2025/Intro to ML/Github/Intro-to-ML/Assignment 2/Housing.csv'
output_path = 'C:/Users/lolze/Documents/Spring 2025/Intro to ML/Github/Intro-to-ML/Assignment 2/Housing2.csv'

#Creates a new file that takes the yes, no data and quantizes it into binary data
with open(file_path, 'r', newline='') as infile, open(output_path, 'w', newline='') as outfile: 
    reader = csv.reader(infile)
    writer = csv.writer(outfile)
    columnindi = [5, 6, 7, 8, 9, 11]
    for row in reader:
        new_row = []
        for i, cell in enumerate(row):
                if i in columnindi:
                    if cell == 'yes':
                        new_cell = cell.replace('yes', '1')
                    if cell == 'no':
                        new_cell = cell.replace('no', '0')
                else:
                    new_cell = cell
                new_row.append(new_cell)
        writer.writerow(new_row)

data = pd.read_csv(output_path)

#sepeate data into features and Y
X = data.iloc[:, 1:12].values
Y = data.iloc[:, 0].values

#split data into train and test 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size = 0.20, random_state=1)

import seaborn as sns
sns.kdeplot(y_train, label='Train')
sns.kdeplot(y_test, label='Test')

#Problem 1

#Develop linear regression with gradient decent algorithm 
def cost_function(theta, X, y): 
    m = len(y)
    prediction = X.dot(theta)
    error = np.subtract(prediction,y)
    sqrerror = np.square(error)
    cost = 1/(2*m) * np.sum(sqrerror)
    return cost 

def gradient_descent(xtrain,xtest,ytrain,ytest, learning_rate, iterations, theta, intercept=False,lasso=False, lambda_=0.1):
    myX = xtrain
    myXtest = xtest
    #factors in if the data has already added the intercept in preprocessing 
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
        gradients = myX.transpose().dot(errors)
        #in the case of regularization
        if lasso == True:
            gradients[1:] += lambda_*np.sign(theta[1:])
            theta = np.subtract(theta,(learning_rate/m)*gradients)
            train_cost = lasso_cost_function(theta,myX,ytrain, lambda_)
            test_cost = lasso_cost_function(theta,myXtest,ytest,lambda_)
        else:
            theta = np.subtract(theta,(learning_rate/m)*gradients)
            train_cost = cost_function(theta,myX,ytrain)
            test_cost = cost_function(theta,myXtest,ytest) 
        cost_history_train.append(train_cost)
        cost_history_test.append(test_cost)
    return theta, cost_history_train,cost_history_test

# evaluate based on area, bedrooms, bathrooms, stories, parking 
#iterations1 = 22 #to avoid integer overflow
iterations1 = 50
lr1 = 0.1
#lr1 =0.01
all_features = ['Area', 'bedrooms', 'bathrooms', 'stories', 'mainroad', 'guestroom', 'basement', 'hotwaterheat', 'aircondition', 'parking', 'prefarea', 'furnishingstatus']
oneafeatures = ['bedrooms', 'bathrooms', 'stories', 'parking']
#oneafeatures = ['Area', 'bedrooms', 'bathrooms', 'stories', 'parking']
def featurelist(currentfeatures):
    featureindex = []
    all_features = ['Area', 'bedrooms', 'bathrooms', 'stories', 'mainroad', 'guestroom', 'basement', 'hotwaterheat', 'aircondition', 'parking', 'prefarea', 'furnishingstatus']
    for x in currentfeatures:
        featureindex.append(all_features.index(x))
    return featureindex
X1a = X_train[:, featurelist(oneafeatures)].astype('float64')
X1a_test = X_test[:, featurelist(oneafeatures)].astype('float64')
theta_1a = np.zeros(len(oneafeatures) + 1)
theta_1a, cost_history1a, cost_history1a_test = gradient_descent(X1a,X1a_test, y_train.astype('float64'), y_test.astype('float64'),lr1,iterations1,theta_1a,intercept=True)
#print(cost_history1a)

plt.figure()
plt.plot(np.linspace(0,iterations1,iterations1), cost_history1a, label='Train')
plt.plot(np.linspace(0,iterations1,iterations1), cost_history1a_test, label='Test')
train1a = cost_history1a[-1]
test1a = cost_history1a_test[-1]
plt.axhline(train1a, color='blue', linestyle=':', alpha=0.3)
plt.axhline(test1a, color='orange', linestyle=':', alpha=0.3)
plt.text(iterations1*1.05, train1a, 
         f'Final Train: {train1a:.2e}', 
         color='blue', va='center')
plt.text(iterations1*1.05,test1a,
         f'Final Test: {test1a:.2e}', 
          color='orange', va='center')
plt.xlabel('Iterations')
plt.ylabel('Loss')
#plt.title('Training and Validation Loss with Area (1a)')
plt.title('Training and Validation Loss (1a)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


#Explore values for learning rate
#Evaluate based on Area, bedrooms, bathrooms, stories, main road, duestroom, basement, hot water heating, air conditioning, parking, prefarea?
onebfeatures = ['bedrooms', 'bathrooms', 'stories', 'mainroad', 'guestroom', 'basement', 'hotwaterheat', 'aircondition', 'parking', 'prefarea']
#onebfeatures = ['Area','bedrooms', 'bathrooms', 'stories', 'mainroad', 'guestroom', 'basement', 'hotwaterheat', 'aircondition', 'parking', 'prefarea']
X1b = X_train[:, featurelist(onebfeatures)].astype('float64')
X1b_test = X_test[:, featurelist(onebfeatures)].astype('float64')
theta_1b = np.zeros(len(onebfeatures)+ 1)
theta_1b, cost_history1b, cost_history1b_test = gradient_descent(X1b, X1b_test, y_train.astype('float64'), y_test.astype('float64'), lr1, iterations1, theta_1b, intercept=True)
#print(cost_history1b)

plt.figure()
plt.plot(np.linspace(0,iterations1,iterations1), cost_history1b, label='Train')
plt.plot(np.linspace(0,iterations1,iterations1), cost_history1b_test, label='Test')
train1b = cost_history1b[-1]
test1b = cost_history1b_test[-1]
plt.axhline(train1b, color='blue', linestyle=':', alpha=0.3)
plt.axhline(test1b, color='orange', linestyle=':', alpha=0.3)
plt.text(iterations1*1.05, train1b, 
         f'Final Train: {train1b:.2e}', 
         color='blue', va='center')
plt.text(iterations1*1.05,test1b,
         f'Final Test: {test1b:.2e}', 
          color='orange', va='center')
plt.xlabel('Iterations')
plt.ylabel('Loss')
#plt.title('Training and Validation Loss Area (1b)')
plt.title('Training and Validation Loss (1b)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

#Problem 2

#normalize data 
from sklearn import preprocessing

sc = preprocessing.MinMaxScaler()
X_train_norm = sc.fit_transform(X_train)
X_test_norm = sc.transform(X_test)
y_train_norm = sc.fit_transform(y_train.reshape(-1, 1)).flatten() 
y_test_norm = sc.transform(y_test.reshape(-1, 1)).flatten() 

#standardize data
sc2 = preprocessing.StandardScaler()
X_train_stand = sc2.fit_transform(X_train)
X_test_stand = sc2.transform(X_test)
y_train_stand = sc2.fit_transform(y_train.reshape(-1, 1)).flatten() 
y_test_stand = sc2.transform(y_test.reshape(-1, 1)).flatten() 

#Normalized: evaluate based on area, bedrooms, stories, parking
#iterations2 = 50
iterations2 = 100
#lr2 = 0.01 test 1
#lr2 = 0.05 Test 2
lr2 = 0.03
twoafeatures = ['Area','bedrooms', 'bathrooms', 'stories', 'parking']
X2a_norm = X_train_norm[:, featurelist(twoafeatures)].astype('float64')
X2a_test_norm = X_test_norm[:, featurelist(twoafeatures)].astype('float64')
theta_2a_norm = np.zeros(len(twoafeatures)+ 1)
theta_2a_norm, cost_history2a_norm, cost_history2a_test_norm = gradient_descent(X2a_norm, X2a_test_norm, y_train_norm.astype('float64'), y_test_norm.astype('float64'), lr2, iterations2, theta_2a_norm, intercept=True)

#Standardized: evaluate based on area, bedrooms, stories, parking
X2a_stand = X_train_stand[:, featurelist(twoafeatures)].astype('float64')
X2a_test_stand = X_test_stand[:, featurelist(twoafeatures)].astype('float64')
theta_2a_stand = np.zeros(len(twoafeatures)+ 1)
theta_2a_stand, cost_history2a_stand, cost_history2a_test_stand = gradient_descent(X2a_stand, X2a_test_stand, y_train_stand.astype('float64'), y_test_stand.astype('float64'), lr2, iterations2, theta_2a_stand, intercept=True)

#plot training and validation losses
plt.figure()
plt.plot(np.linspace(0,iterations2,iterations2), cost_history2a_norm, label='Train')
plt.plot(np.linspace(0,iterations2,iterations2), cost_history2a_test_norm, label='Test')
train2a = cost_history2a_norm[-1]
test2a = cost_history2a_test_norm[-1]
plt.axhline(train2a, color='blue', linestyle=':', alpha=0.3)
plt.axhline(test2a, color='orange', linestyle=':', alpha=0.3)
plt.text(iterations2*1.05, train2a, 
         f'Final Train: {train2a:.2e}', 
         color='blue', va='center')
plt.text(iterations2*1.05,test2a,
         f'Final Test: {test2a:.2e}', 
          color='orange', va='center')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.title('Training and Validation Loss Normalized Data (2a)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


plt.figure()
plt.plot(np.linspace(0,iterations2,iterations2), cost_history2a_stand, label='Train')
plt.plot(np.linspace(0,iterations2,iterations2), cost_history2a_test_stand, label='Test')
train2as = cost_history2a_stand[-1]
test2as = cost_history2a_test_stand[-1]
plt.axhline(train2as, color='blue', linestyle=':', alpha=0.3)
plt.axhline(test2as, color='orange', linestyle=':', alpha=0.3)
plt.text(iterations2*1.05, train2as, 
         f'Final Train: {train2as:.2e}', 
         color='blue', va='center')
plt.text(iterations2*1.05,test2as,
         f'Final Test: {test2as:.2e}', 
          color='orange', va='center')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.title('Training and Validation Loss Standardized Data (2a)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

#TODO compare scaling approaches an which achieves the best training 
#Normalized: Evaluate based on Area, bedrooms, bathrooms, stories, main road, duestroom, basement, hot water heating, air conditioning, parking, prefarea?
twobfeatures = ['Area','bedrooms', 'bathrooms', 'stories', 'mainroad', 'guestroom', 'basement', 'hotwaterheat', 'aircondition', 'parking', 'prefarea']
X2b_norm = X_train_norm[:, featurelist(twobfeatures)].astype('float64')
X2b_test_norm = X_test_norm[:, featurelist(twobfeatures)].astype('float64')
theta_2b_norm = np.zeros(len(twobfeatures)+ 1)
theta_2b_norm, cost_history2b_norm, cost_history2b_test_norm = gradient_descent(X2b_norm, X2b_test_norm, y_train_norm.astype('float64'), y_test_norm.astype('float64'), lr2, iterations2, theta_2b_norm, intercept=True)

#Standardized: Evaluate based on Area, bedrooms, bathrooms, stories, main road, duestroom, basement, hot water heating, air conditioning, parking, prefarea?
X2b_stand = X_train_norm[:, featurelist(twobfeatures)].astype('float64')
X2b_test_stand = X_test_norm[:, featurelist(twobfeatures)].astype('float64')
theta_2b_stand = np.zeros(len(twobfeatures)+ 1)
theta_2b_stand, cost_history2b_stand, cost_history2b_test_stand = gradient_descent(X2b_stand, X2b_test_stand, y_train_stand.astype('float64'), y_test_stand.astype('float64'), lr2, iterations2, theta_2b_stand, intercept=True)

#plot training and validation losses
plt.figure()
plt.plot(np.linspace(0,iterations2,iterations2), cost_history2b_norm, label='Train')
plt.plot(np.linspace(0,iterations2,iterations2), cost_history2b_test_norm, label='Test')
train2b = cost_history2b_norm[-1]
test2b = cost_history2b_test_norm[-1]
plt.axhline(train2b, color='blue', linestyle=':', alpha=0.3)
plt.axhline(test2b, color='orange', linestyle=':', alpha=0.3)
plt.text(iterations2*1.05, train2b, 
         f'Final Train: {train2b:.3e}', 
         color='blue', va='center')
plt.text(iterations2*1.05,test2b,
         f'Final Test: {test2b:.3e}', 
          color='orange', va='center')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.title('Training and Validation Loss Normalized Data (2b)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

plt.figure()
plt.plot(np.linspace(0,iterations2,iterations2), cost_history2b_stand, label='Train')
plt.plot(np.linspace(0,iterations2,iterations2), cost_history2b_test_stand, label='Test')
train2bs = cost_history2b_stand[-1]
test2bs = cost_history2b_test_stand[-1]
plt.axhline(train2bs, color='blue', linestyle=':', alpha=0.3)
plt.axhline(test2bs, color='orange', linestyle=':', alpha=0.3)
plt.text(iterations2*1.05, train2bs, 
         f'Final Train: {train2bs:.2e}', 
         color='blue', va='center')
plt.text(iterations2*1.05,test2bs,
         f'Final Test: {test2bs:.2e}', 
          color='orange', va='center')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.title('Training and Validation Loss Standardized Data (2b)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

#Problem 3 
#TODO Repeat 2a with parameter penalties (make sure to modify gradient descent logic on training set)) 

def lasso_cost_function(theta,X, y, lambda_):
    mse = cost_function(theta,X,y) 
    penalty = lambda_*np.sum(np.abs(theta[1:]))
    return mse+penalty


#Lambda Exploration
lambdas = np.logspace(-3, 1, 20)
final_cost = []
for val in lambdas: 
    theta_3a = np.zeros(len(twoafeatures)+ 1)
    theta_3a, cost_history3a, cost_history3a_test = gradient_descent(X2a_norm, X2a_test_norm, y_train_norm.astype('float64'), y_test_norm.astype('float64'), lr2, iterations2, theta_3a, intercept=True, lasso=True, lambda_=val)
    final_cost.append(cost_history3a_test[-1])
best_lambda = lambdas[4]

plt.figure()
plt.plot(lambdas, final_cost, marker='o')
plt.axvline(best_lambda, color='red', linestyle='--', alpha=0.7, linewidth=1.5)
plt.text(best_lambda+2, 1,  
         f'Optimal λ = {best_lambda:.2e}\nLoss = {final_cost[4]:.2e}',
         color='red', ha='center', bbox=dict(facecolor='white', alpha=0.8))
plt.xlabel('Lambda')
plt.ylabel('Validation Loss')
plt.title('Lasso Validation Loss vs. Lambda Values (3a)')
plt.grid()
plt.tight_layout
plt.show()
#print(lambdas)
#print(final_cost)

theta_3a, cost_history3a, cost_history3a_test = gradient_descent(X2a_norm, X2a_test_norm, y_train_norm.astype('float64'), y_test_norm.astype('float64'), lr2, iterations2, theta_3a, intercept=True, lasso=True, lambda_=best_lambda)
#print(theta_3a)
#print(theta_2a_norm)
#PLot results 
plt.figure()
plt.plot(np.linspace(0,iterations2,iterations2), cost_history3a, label='Train')
plt.plot(np.linspace(0,iterations2,iterations2), cost_history3a_test, label='Test')
train3a = cost_history3a[-1]
test3a = cost_history3a_test[-1]
plt.axhline(train3a, color='blue', linestyle=':', alpha=0.3)
plt.axhline(test3a, color='orange', linestyle=':', alpha=0.3)
plt.text(iterations2*1.05, train3a, 
         f'Final Train: {train3a:.2e}', 
         color='blue', va='center')
plt.text(iterations2*1.05,test3a,
         f'Final Test: {test3a:.2e}', 
          color='orange', va='center')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.title('Training and Validation Loss Normalized Data with Lasso Regression (3a)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

#TODO Repeat 2b with parameter penalties
lambdas = np.logspace(-3, 1, 20)
final_cost = []
for val in lambdas: 
    theta_3b = np.zeros(len(twobfeatures)+ 1)
    theta_3b, cost_history3b, cost_history3b_test = gradient_descent(X2b_norm, X2b_test_norm, y_train_norm.astype('float64'), y_test_norm.astype('float64'), lr2, iterations2, theta_3b, intercept=True, lasso=True, lambda_=val)
    final_cost.append(cost_history3b_test[-1])

best_lambda = lambdas[3]
plt.figure()
plt.plot(lambdas, final_cost, marker='o')
plt.axvline(best_lambda, color='red', linestyle='--', alpha=0.7, linewidth=1.5)
plt.text(best_lambda+2, 2,  
         f'Optimal λ = {best_lambda:.2e}\nLoss = {final_cost[3]:.2e}',
         color='red', ha='center', bbox=dict(facecolor='white', alpha=0.8))
plt.xlabel('Lambda')
plt.ylabel('Validation Loss')
plt.title('Lasso Validation Loss vs. Lambda Values (3b)')
plt.grid()
plt.tight_layout()
plt.show()
#print(lambdas)
#print(final_cost)

theta_3b, cost_history3b, cost_history3b_test = gradient_descent(X2b_norm, X2b_test_norm, y_train_norm.astype('float64'), y_test_norm.astype('float64'), lr2, iterations2, theta_3b, intercept=True, lasso=True, lambda_=best_lambda)
#print(theta_3b)
#print(theta_2b_norm)
#PLot results 
plt.figure()
plt.plot(np.linspace(0,iterations2,iterations2), cost_history3b, label='Train')
plt.plot(np.linspace(0,iterations2,iterations2), cost_history3b_test, label='Test')
train3b = cost_history3b[-1]
test3b = cost_history3b_test[-1]
plt.axhline(train3b, color='blue', linestyle=':', alpha=0.3)
plt.axhline(test3b, color='orange', linestyle=':', alpha=0.3)
plt.text(iterations2*1.05, train3b, 
         f'Final Train: {train3b:.2e}', 
         color='blue', va='center')
plt.text(iterations2*1.05,test3b,
         f'Final Test: {test3b:.2e}', 
          color='orange', va='center')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.title('Training and Validation Loss Normalized Data with Lasso Regression (3b)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()