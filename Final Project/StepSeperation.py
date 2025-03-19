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
    
'''