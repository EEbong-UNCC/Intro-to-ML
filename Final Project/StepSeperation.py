import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import csv

#TODO Seperate out into steps

'''
Logic: A step is defined as the heel strike of one foot to the heel strike of the opposite foot. In our data set, we do not have a standardized starting foot. Therefore in order to count number of steps and seperate it out into those steps

Seperation: Our program needs to detect which foot is the first to strike. We define this as our slope going from near zero and increasing. The step lasts until the opposite foot has this same step increase in slope

Pseudocode: 

calculate gradient of FP1z, FP2z 
when gradient increasing from 0 
    note index in array heel strike

compare FP1z_heel_strike[0] to FP2z_heel_strike[0]
if FP1z_heel_strike[0] < FP2z_heel_strike[0]
    Step start index[0][0] = FP1z_heel_strike[0]
    Step end index[0][1] = FP2z_heel_strike[0]
else 
    Step start index [0][0] = FP2z_heel_strike[0]
    Step end index [0][1] = FP1z_heel strike[0]
    
fill in the rest of the values from there 

Purpose:

In order to extract average force per step, stance duration, swing duration, time to peak force, etc, we need to know each step   

- further features
    - step duration
    - average loading rate ( (peak force - initial force)/time to peak force) 
    - average unloading rate (peak force - final force/ time from peak force to toe-off) 
    - Double Support time (time between heel strike of one foot and toe off on the other (aka time from 0 -> + slope for one foot to other foot force just reaching 0) 
    
put steps into array 
put array in array 
dataframe of arrays
'''
def gradient(instance): 
    left = instance['FP1_z']
    right = instance['FP2_z']
    heel_strike_index = [[],[]]
    row_num = np.linspace(0,len(left), len(left))
    left_slope = np.gradient(left, row_num)
    right_slope = np.gradient(right, row_num)
    for x in range(len(left_slope)):
        if left_slope[x] == 0:
            if left_slope[x+1] != 0:
                heel_strike_index[0].append(x)
        if right_slope[x] == 0:
            if right_slope[x+1] !=0:
                heel_strike_index[1].append(x)
    return heel_strike_index

#have an array with indices that is a sliding window 
def first_step(heel_strike_index): 
    i = 0
    steps = []
    #if left foot strikes first
    if heel_strike_index[0] < heel_strike_index[1]:
        combined = zip(heel_strike_index[:,0], heel_strike_index[:,1])
    if heel_strike_index[1] < heel_strike_index[0]:
        combined = zip(heel_strike_index[:,0], heel_strike_index[:,1])
    for x, y in combined:
        steps.append(x)
        steps.append(y)
    return steps