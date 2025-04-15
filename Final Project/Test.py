import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import csv
from stepseperation import *
from featureextraction import *
from featuredocument import *

force_data = pd.read_csv('C:/Users/lolze/Documents/Spring 2025/Intro to ML/Github/Intro-to-ML/Final Project/Data/GP1_0.7_force.csv')

an = process_instance(1, force_data, 10)


#Testing the Functions in FeatureExtraction and Step Separation

# Calculate the slope of GRF
force_plate_1 = force_data["FP1_z"]
force_plate_2 = force_data["FP2_z"]
row_num = np.linspace(0,len(force_plate_1), len(force_plate_1))
force_data["slope_FP1"] = np.gradient(force_plate_1, row_num)
force_data["slope_FP2"] = np.gradient(force_plate_2, row_num)
heels = strike_gradient(force_data,10)
toes = lift_gradient(force_data, 10)
mysteps = first_step(heels)
stance = stance_extraction(heels, toes)
firsrfoot = first_foot(heels, mysteps)
forces = ['FP1_x','FP2_x', 'FP1_y', 'FP2_y', 'FP1_z', 'FP2_z']
pf = peak_force(force_data, forces)
mf = mean_force(force_data, stance, forces)
lmfti = mean_FTI(stance[0], force_plate_1)
rmfti = mean_FTI(stance[1], force_plate_2)
steptime = step_duration(mysteps)
lr, ulr, ttp = peakandloading(force_data, stance)
doubs = double_support(heels, toes, firsrfoot)
std, swd = ssduration(stance)

