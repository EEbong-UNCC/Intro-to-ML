import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv

#I should normalize data after i create the dataset not before dumb dumb dumb
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
forces = ['FP1_x','FP2_x', 'FP1_y', 'FP2_y', 'FP1_z', 'FP2_z']
for key in data: 
    instance = data[key]
    for x in range(6): 
        inst_force = instance[forces[x]]
        if max(inst_force) > max_forces[x]:
            max_forces[x] = max(inst_force)
        if min(inst_force) < min_forces[x]: 
            min_forces[x] = min(inst_force)

#TODO create a new csv for each instance, post normalization where each value is replaced by the value - min value/(max value - min value)

for key in data: 
    instance = data[key]
    for var_name in forces: 
        for index in range(len(instance[var_name])):
            value = instance[var_name][index]
            norm_val = (value - min_forces[forces.index(var_name)])/(max_forces[forces.index(var_name)] - min_forces[forces.index(var_name)])
            instance[var_name][index] = norm_val
    file_name = 'C:/Users/lolze/Documents/Spring 2025/Intro to ML/Github/Intro-to-ML/Final Project/Normalized Data/' + key + 'norm.csv'
    with open(file_name, 'w', newline='') as f: 
        writer = csv.writer(f)
        writer.writerow(forces)
        for i in range(len(instance[forces[0]])):

            row = [instance[var_name][i] for var_name in forces]
            writer.writerow(row)
