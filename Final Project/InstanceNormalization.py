import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#TODO Import Data in such a way that the file name can be changed
data = {} 
instance_speeds = [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7]
for sub in range(1, 23):
    if sub == 4: 
        continue
    path_base = 'C:/Users/lolze/Documents/Spring 2025/Intro to ML/Github/Intro-to-ML/Final Project/Data/'
    for speed in instance_speeds:
        name = 'GP' + str(sub) + '_'+ str(speed) + '_force'
        path = path_base + name + '.csv'
        df = pd.read_csv(path)
        data[name] = df

#TODO for each data set, scan each value, find the max and the min value and save them 
max_forces = [0, 0, 0, 0, 0, 0]
min_forces = [0, 0, 0, 0, 0, 0]
for key in data: 

#TODO create a new csv for each instance, post normalization where each value is replaced by the value - min value/(max value - min value)