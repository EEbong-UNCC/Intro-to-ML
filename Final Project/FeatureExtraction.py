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


#extract peak for for x, y , z for each plate (6 features)
#index [1-6]
forces = ['FP1_x','FP2_x', 'FP1_y', 'FP2_y', 'FP1_z', 'FP2_z']

#input instance, output peak_force 6 features
def peak_force(instance, forces):
    max_forces = np.zeros(6)
    for ind, ele in enumerate(forces): 
        inst_force = instance[ele]
        max_forces[ind] = max(inst_force)
    peak_force.append(max_forces)
    return max_forces

#mean force x, y, z (6 features)
'''
Mean Force Extraction from determine which foot starts get from function first_foot
for the stance phase per foot we are going from heel strike to heel lift. 
However a step is from heel step to heel step neglecting the double support time.
we assume that stances have already been extracted
def mean force extracion(steps, step_stances, column):
    mean force = []
    for x in step_stances
        mean force.append(np.mean(steps[x[0]:x[1],column])
    return np.mean(meanforce)

meanforceextraction(steps, left_stances)
meanforceextraction(steps, right_stances)

Index 7-12
'''
#input instance, output mean force 6 features
def mean_force(instance, stances, forces):
    mean_force = [[],[],[],[],[],[]]
    for x in stances:
        for index,force in enumerate(forces):
            step_data = instance[x[0]:x[1], force]
            mean_force[index].append(np.mean(step_data))
    for index, arr in enumerate(mean_force):
        mean_force[index] = np.mean(mean_force)
    return mean_force

#go through each data point 
#find max of each feature and number of participant 
#write to a new file called Feautre_Peak_Force.csv
#force time integral (1 feature)
'''
Only do for z forces 
High FTI indicates longer stances or higher forces
Lower FTI suggest lighter or quicker steps    
'''
#index 14
from scipy.integrate import simps
def mean_FTI(stances,FP1_z):
    answer = []
    for x in stances:
        time = np.arange(x[0], x[1] + 1)/1000
        force = FP1_z[x[0], x[1] + 1]
        fti = simps(force, time)
        answer.append(fti)
    return np.mean(answer)

#TODO step duration
#TODO average loading rate ( (peak force - initial force)/time to peak force) 
#TODO average unloading rate (peak force - final force/ time from peak force to toe-off) 
#TODO Double Support time (time between heel strike of one foot and toe off on the other (aka time from 0 -> + slope for one foot to other foot force just reaching 0) 
#TODO gait asymmetry compare slope of left and right
#TODO Swing Duration 
#TODO Stance Duration
#TODO Time to Peak Force 