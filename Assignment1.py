#Eka Ebong

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Data is X1, X2, X3 and Y
data = pd.read_csv('C:/Users/lolze/Documents/Spring 2025/Intro to ML/D3.csv')

"""
Problem 1
Linear Regression with Gradient descent algorithm for each explanatory variable in isolation
Assume only one explanatory variable is explaining the output
3 different trainings for each x
Learning rate: Explore between 0.01 and 0.1
Theta_initial = 0
"""
#Implement cost function
def cost_function(theta, X, y):
    m = len(y)
    predictions = X.dot(theta)
    cost = (1/(2*m)) * np.sum(predictions-y)**2
    return cost


#Implement gradient Descent
def gradient_descent(X, y, learning_rate, iterations, theta):
    m = len(y)
    cost_history = []

    for i in range(iterations):
        gradients = (1/m)*X.T.dot(X.dot(theta) - y)
        theta -= learning_rate* gradients
        cost_history.append(cost_function(theta,X,y))
    return theta, cost_history

#Function to run 3 times
variables = ['X1', 'X2', 'X3']
learning_rate_test = [ 0.01, 0.03, 0.05, 0.07, 0.1]

#Plot Exploration of Different Learning Rates for each variable
for feature in range(len(variables)):
    plt.figure()
    print(variables[feature])
    for test_rate in range(len(learning_rate_test)):
        theta = np.zeros(2)
        x1 = np.asarray(data[variables[feature]])
        x = np.c_[np.ones((100, 1)), x1]
        y = np.asarray(data['Y'])
        cost_function(theta, x, y)
        theta, cost_history = gradient_descent(x, y, learning_rate_test[test_rate], 50, theta)
        plt.plot(range(len(cost_history)), cost_history, label=f'Learning Rate: {learning_rate_test[test_rate]}')
        print(learning_rate_test[test_rate])
    plt.xlabel('Iterations')
    plt.ylabel('Cost Function')
    plt.title(f'Cost Function over Iteration with Feature {variables[feature]}')
    plt.legend()
    plt.grid()
    plt.show()

final_lr = [0.1, 0.03, 0.07]

#Plot Final Regressions for chosen Alpha values
for feature in range(len(variables)):
    plt.figure()
    theta = np.zeros(2)
    x1 = np.asarray(data[variables[feature]])
    x = np.c_[np.ones((100, 1)), x1]
    y = np.asarray(data['Y'])
    cost_function(theta, x, y)
    theta, cost_history = gradient_descent(x, y, final_lr[feature], 50, theta)
    print("Final Linear Model: y = {:.2f} + {:.2f} * x".format(theta[0], theta[1],))

    #Plot Final Regression
    plt.plot(x1, (theta[0] + x1*theta[1]), color='red')
    plt.plot(x1, y, 'o', color='blue', markersize=8)
    plt.xlabel('X')
    plt.ylabel('Prediction')
    plt.title(f'Linear Regression for {variables[feature]}')
    plt.legend()
    plt.grid()
    plt.show()

for feature in range(len(variables)):
    plt.figure()
    print(variables[feature])
    theta = np.zeros(2)
    x1 = np.asarray(data[variables[feature]])
    x = np.c_[np.ones((100, 1)), x1]
    y = np.asarray(data['Y'])
    cost_function(theta, x, y)
    theta, cost_history = gradient_descent(x, y, final_lr[feature], 50, theta)
    plt.plot(range(len(cost_history)), cost_history, label=f'Learning Rate: {final_lr[feature]}')
    plt.xlabel('Iterations')
    plt.ylabel('Cost Function')
    plt.title(f'Cost Function over Iteration with Feature {variables[feature]}')
    plt.legend()
    plt.grid()
    plt.show()

"""
Problem 2
Run a Linear Regression with gradient descent using all 3 variables
Learning Rate: Explore between 0.01 and 0.1 (your choice)
"""

#Implement function with 3 variables
#TODO PLOT Loss

x_all = data.iloc[:, :3]
x_all = np.c_[np.ones((100,1)), x_all]
alphas = [0.01, 0.03, 0.05, 0.07, 0.1]
for a in range(len(alphas)):
    theta = np.zeros(4)
    theta, cost_history = gradient_descent(x_all, y, alphas[a], 50, theta)
    # Report final linear model
    print("Final Linear Model: y = {:.2f} + {:.2f} * x1 + {:.2f} * x2 + {:.2f} * x3".format(theta[0], theta[1], theta[2], theta[3]))

    # Plot loss over iterations
    plt.plot(range(len(cost_history)), cost_history, label=f'Learning Rate: {alphas[a]}')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.title(f'Cost Function over Iteration With 3 Variables')
plt.grid()
plt.legend()
plt.show()

# Predict new values
theta = np.zeros(4)
theta, cost_history = gradient_descent(x_all, y, 0.05, 50, theta)
plt.plot(range(len(cost_history)), cost_history, label=f'Learning Rate: {0.05}')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.title(f'Cost Function over Iteration With 3 Variables')
plt.grid()
plt.legend()
plt.show()

new_data = np.array([[1, 1, 1], [2, 0, 4], [3, 2, 1]])
new_data = np.c_[np.ones((3,1)), new_data]  # Add intercept term
predictions = new_data.dot(theta)
print("Predictions for new data:", predictions)
print("Final Linear Model: y = {:.2f} + {:.2f} * x1 + {:.2f} * x2 + {:.2f} * x3".format(theta[0], theta[1], theta[2], theta[3]))
