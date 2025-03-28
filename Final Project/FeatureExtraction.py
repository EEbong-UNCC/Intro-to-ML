import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import csv
data = {} 
instance_speeds = [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7]
for sub in range(1, 23):
    if sub == 4: 
        continue
    path_base = 'C:/Users/lolze/Documents/Spring 2025/Intro to ML/Github/Intro-to-ML/Final Project/Normalized Data/'
    for speed in instance_speeds:
        name = 'GP' + str(sub) + '_'+ str(speed) + '_force'
        path = path_base + name + 'norm.csv'
        df = pd.read_csv(path)
        data[name] = df

#TODO extract peak for for x, y , z for each plate (6 features)
peak_force = []
forces = ['FP1_x','FP2_x', 'FP1_y', 'FP2_y', 'FP1_z', 'FP2_z']
for key in data: 
    instance = data[key]
    max_forces = np.zeros(6)
    for x in range(6): 
        inst_force = instance[forces[x]]
        max_forces[x] = max(inst_force)
    peak_force.append(max_forces)

print(peak_force)
#go through each data point 
#find max of each feature and number of participant 
#write to a new file called Feautre_Peak_Force.csv
#TODO force time integral (1 feature)
#TODO mean force x, y, z (6 features)
#TODO mean force left/right (3 features)
#TODO gait asymmetry compare slope of left and right