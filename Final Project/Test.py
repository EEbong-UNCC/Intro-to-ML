import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import csv
from StepSeparation import *
from FeatureExtraction import *

force_data = pd.read_csv('C:/Users/lolze/Documents/Spring 2025/Intro to ML/Github/Intro-to-ML/Final Project/Data/GP1_0.6_force.csv')

# Calculate the slope of GRF
force_plate_1 = force_data["FP1_z"]
force_plate_2 = force_data["FP2_z"]
row_num = np.linspace(0,len(force_plate_1), len(force_plate_1))
force_data["slope_FP1"] = np.gradient(force_plate_1, row_num)
force_data["slope_FP2"] = np.gradient(force_plate_2, row_num)
heels = strike_gradient(force_data,10)
toes = lift_gradient(force_data, 10)
mysteps = first_step(heels)
leftstance, rightstance = stance_extraction(heels, toes)
firsrfoot = first_foot(heels, mysteps)

#TODO test functionality of feature extraction functions
