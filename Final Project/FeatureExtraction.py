import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import csv
from stepseperation import *

#input instance, output peak_force 6 features
def peak_force(instance, forces):
    max_forces = np.zeros(6)
    for ind, ele in enumerate(forces): 
        inst_force = instance[ele]
        max_forces[ind] = max(inst_force)
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
        stance = stances[(index)%2]
        for x in stance:
            step_data = instance.iloc[x[0]:x[1], forces.index(force)]
            mean_force[index].append(np.mean(step_data))
        mean_force[index] = np.mean(np.array(mean_force[index]))
    return mean_force

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
        force = FP1_z[x[0]:x[1] + 1]
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
    return np.mean(duration)/1000
 
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
    for index, stance in enumerate(stances):
        for i in stance:
           stance_start = i[0]
           stance_end = i[1]
           peak_force = np.max(instance[column[index]][stance_start:stance_end])
           peak_force_time = instance.index[instance[column[index]]==peak_force][0]
           if stance_start > peak_force_time or stance_end < peak_force_time:
               peak_force_time = instance.index[instance[column[index]]==peak_force][1]
               #TODO Figure out why this is triggering
           #ttpf: time to peak force
           ttpf = (peak_force_time - stance_start)/1000
           time_to_peak[index].append(ttpf)
           #lr: loading rate
           lr = (peak_force - instance[column[index]][stance_start])/ttpf
           loading_rate[index].append(lr)
           #tfpf: time from peak force
           tfpf = (stance_end - peak_force_time)/1000
           ulr = (peak_force - instance[column[index]][stance_end])/tfpf
           unloading_rate[index].append(ulr)
        loading_rate[index] = np.mean(np.array(loading_rate[index]))
        unloading_rate[index] = np.mean(np.array(unloading_rate[index]))
        time_to_peak[index] = np.mean(np.array(time_to_peak[index]))
    return loading_rate, unloading_rate, time_to_peak   
            
#Double Support time (time between heel strike of one foot and toe off on the other)
# index 22, 23
def double_support(heel_strike_index, toe_off_index, first_foot):
    double = [] 
    feet = ['left','right']
    firstind = feet.index(first_foot)
    #pick the shorter array as index
    length = min( [ len(toe_off_index[firstind-1]),len(heel_strike_index[firstind]) ])
    for x in range(length):
        time = toe_off_index[firstind-1][x] - heel_strike_index[firstind][x]
        double.append(time/1000)
    arr = np.where(np.array(toe_off_index[firstind]) < heel_strike_index[firstind-1][0], True, False)
    startind = arr
    for index, object in enumerate(arr): 
        if object == False: 
            startind = index
            break
    #pick the shorter array as index
    length = min([ len(toe_off_index[firstind])-startind,len(heel_strike_index[firstind-1]) ])
    for x in range(length):
        time = toe_off_index[firstind][startind+x] - heel_strike_index[firstind-1][x]
        double.append(time/1000)
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
            x = inst[0]
            y = inst[1]
            swing_duration[i].append(np.abs(x-toe_off))
            stance_duration[i].append(y-x)
            toe_off = y
        stance_duration[i] = np.mean(stance_duration[i])/1000
        swing_duration[i] = np.mean(swing_duration[i])/1000    
        i += 1
    return np.mean(stance_duration), np.mean(swing_duration)
    
def process_instance(y, instance, threshold):
    answer = []
    answer.append(y)
    forces = ['FP1_x','FP2_x', 'FP1_y', 'FP2_y', 'FP1_z', 'FP2_z']
    #Pre-processing 
    heels = strike_gradient(instance, threshold)
    toes = lift_gradient(instance, threshold)
    mysteps = first_step(heels)
    stance = stance_extraction(heels, toes)
    firstfoot = first_foot(heels, mysteps)

    #Peak Forces 6
    pf = peak_force(instance, forces)
    answer.extend(pf)

    #Mean Forces 6
    mf = mean_force(instance, stance, forces)
    answer.extend(mf)

    #Mean FTI
    fti1 = mean_FTI(stance[0], instance[forces[4]])
    fti2 = mean_FTI(stance[1], instance[forces[5]])
    answer.append(fti1)
    answer.append(fti2)

    #Step Duration 
    sd = step_duration(mysteps)
    answer.append(sd)

    #Average Loading Rate Left/Right, Average Unloading Rate Left/Right, Time to peak Left/Right
    lr, ulr, tp = peakandloading(instance, stance)
    answer.extend(lr)
    answer.extend(ulr)
    answer.extend(tp)

    #Double Support Time
    ds = double_support(heels, toes, firstfoot)
    answer.append(ds)

    #Swing and Stance Duration
    std, swd = ssduration(stance)
    answer.append(std)
    answer.append(swd)
    
    return answer
#TODO start writing code for the actual models 