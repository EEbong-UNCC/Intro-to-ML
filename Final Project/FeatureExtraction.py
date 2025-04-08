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
    for index,force in enumerate(forces):
        stance = stances[index//2]
        for x in stance:
            step_data = instance[x[0]:x[1], force]
            mean_force[index].append(np.mean(step_data))
    return np.mean(mean_force, axis=1)

#go through each data point 
#find max of each feature and number of participant 
#write to a new file called Feautre_Peak_Force.csv
#force time integral (1 feature)
'''
Only do for z forces 
High FTI indicates longer stances or higher forces
Lower FTI suggest lighter or quicker steps  
Call on stances[0] and stances [1]  
'''
#index 13,14
from scipy.integrate import simps
def mean_FTI(stance,FP1_z):
    answer = []
    for x in stance:
        time = np.arange(x[0], x[1] + 1)/1000
        force = FP1_z[x[0], x[1] + 1]
        fti = simps(force, time)
        answer.append(fti)
    return np.mean(answer)

#step duration index 15
'''
Uses step data just calculate distance between sliding window of 2 and average
'''
def step_duration(steps): 
    duration = []
    for i in range(1,len(steps)): 
        length = steps[i] - steps[i-1]
        duration.append(length)
    return np.mean(duration)
 
'''
Input: instance, stances
Output: average loading rate for left and right foot, average unloading rate for left and right foot and 
    average time to peak force for left and right foot

average loading rate ( (peak force - initial force)/time to peak force) 
average unloading rate (peak force - final force/ time from peak force to toe-off)
'''
#index 16, 17, 18, 19, 20, 21
def peakandloading(instance,stances): 
    column = ['FP1_z', 'FP2_z']
    loading_rate = [[],[]]
    unloading_rate = [[],[]]
    time_to_peak = [[],[]]
    for index, stance in stances:
        for i in stance:
           stance_start = stance[0]
           stance_end = stance[1]
           peak_force = np.max(instance[column[index]][stance_start:stance_end])
           peak_force_time = instance[column[index]].index(peak_force)
           if stance_start > peak_force_time or stance_end < peak_force_time:
               print('Uh OH')
           #ttpf: time to peak force
           ttpf = peak_force_time - stance_start
           time_to_peak[index].append(ttpf)
           #lr: loading rate
           lr = (peak_force - instance[column[index]][stance_start])/ttpf
           loading_rate[index].append(lr)
           #tfpf: time from peak force
           tfpf = stance_end - peak_force_time
           ulr = (peak_force - instance[column[index]][stance_end])/tfpf
           unloading_rate[index].append(ulr)
    
    return np.mean(loading_rate,axis=1), np.mean(unloading_rate,axis=1), np.mean(time_to_peak, axis=1)   
            
#Double Support time (time between heel strike of one foot and toe off on the other)
# index 22, 23
def double_support(heel_strike_index, toe_off_index, first_foot):
    double = [] 
    feet = ['left','right']
    firstind = feet.index(first_foot)
    for x in range(len(heel_strike_index[firstind])):
        time = toe_off_index[firstind-1][x] - heel_strike_index[firstind][x]
        double.append(time)
    arr = toe_off_index[firstind] < heel_strike_index[firstind-1][0]
    startind = 0
    for index, object in arr: 
        if object == False: 
            startind = index
            break
    for x in range(len(toe_off_index[firstind])-startind):
        time = toe_off_index[firstind][startind+x] - heel_strike_index[firstind-1][x]
        double.append(time)
    return np.mean(double)
    
#Swing Duration 
#Stance Duration
# index 24, 25
def ssduration(stances):
    stance_duration = [[],[]]
    swing_duration = [[], []]
    i = 0
    for stance in stances:
        toe_off = 0
        for inst in stance: 
            for x,y in inst:
                swing_duration[i].append(np.abs(x-toe_off))
                stance_duration[i].append(y-x)
                toe_off = y    
        i += 1
    return np.mean(stance_duration, axis=1), np.mean(swing_duration, axis=1)
    

#TODO find all instances of time and divide by 1000 to get the correct time
#TODO full loop for extraction of all features of one instance in new file
#TODO start writing code for the actual models 