import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import csv
from featureextraction import *

#Put data in dataframe
y = []
data = {} 
instance_speeds = [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7]
for sub in range(1, 23):
    if sub == 4: 
        continue
    path_base = 'C:/Users/lolze/Documents/Spring 2025/Intro to ML/Github/Intro-to-ML/Final Project/Data/'
    for speed in instance_speeds:
        y.append(sub)
        name = 'GP' + str(sub) + '_'+ str(speed) + '_force'
        path = path_base + name + '.csv'
        df = pd.read_csv(path)
        data[name] = df

#Create CSV of Extracted Features 
column = ['Y','Peak_Force_X1','Peak_Force_X2','Peak_Force_Y1','Peak_Force_Y2','Peak_Force_Z1','Peak_Force_Z2'
          ,'Mean_Force_X1','Mean_Force_X2','Mean_Force_Y1','Mean_Force_Y2','Mean_Force_Z1','Mean_Force_Z2'
          ,'Mean_FTI_Left', 'Mean_FTI_Right','Step_Duration','Average_Loading_Rate_Left','Average_Loading_Rate_Right'
          ,'Average_Unloading_Rate_Left','Average_Unloading_Rate_Right','Time_To_Peak_Force_Left','Time_To_Peak_Force_Right'
          ,'Double_Support_Time','Stance_Duration','Swing_Duration', 'FFT_Mean_Left', 'FFT_std_Left', 'FFT_Max_Left', 'FFT_Mean_Right'
          , 'FFT_std_Right', 'FFT_Max_Right', 'PSD_Peak_Freq_Left', 'PSD_Total_Power_Left', 'PSD_Band_Ratio_Left', 'PSD_Peak_Freq_Right'
          , 'PSD_Total_Power_Right', 'PSD_Band_Ratio_Right', 'Harmonic_Ratio_Left', 'Harmonic_Ratio_Right']

threshold = 10 
file_name = 'C:/Users/lolze/Documents/Spring 2025/Intro to ML/Github/Intro-to-ML/Final Project/FullGaitFeatures.csv'
with open(file_name, 'w', newline='') as f: 
    writer = csv.writer(f)
    writer.writerow(column)
    for index, instance in enumerate(data.values()):
        row = freqprocess_instance(y[index],instance,threshold)
        writer.writerow(row)
