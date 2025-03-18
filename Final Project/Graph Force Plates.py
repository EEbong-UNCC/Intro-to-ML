import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load force data for Force Plate 1 (left foot) and Force Plate 2 (right foot)
force_data = pd.read_csv('C:/Users/lolze/Documents/Spring 2025/Intro to ML/Github/Intro-to-ML/Final Project/Data/GP1_0.7_force.csv')

# Calculate the slope of GRF
force_plate_1 = force_data["FP1_z"]
force_plate_2 = force_data["FP2_z"]
row_num = np.linspace(0,len(force_plate_1), len(force_plate_1))
force_data["slope_FP1"] = np.gradient(force_plate_1, row_num)
force_data["slope_FP2"] = np.gradient(force_plate_2, row_num)

# Plot GRF for Force Plate 1 (left foot)
plt.subplot(3, 1, 1)
plt.plot(row_num, force_plate_1, label="GRF (Left Foot)") 
plt.xlabel("Time (s)")
plt.ylabel("GRF (N)")
plt.title("GRF and Slope for Force Plate 1 (Left Foot)")
plt.legend()

# Plot GRF for Force Plate 2 (right foot)
plt.subplot(3, 1, 2)
plt.plot(row_num, force_plate_2, label="GRF (Right Foot)")
plt.xlabel("Time (s)")
plt.ylabel("GRF (N)")
plt.title("GRF and Slope for Force Plate 2 (Right Foot)")
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(row_num, force_data["slope_FP1"], label="Slope (Left Foot)")
plt.plot(row_num, force_data["slope_FP2"], label="Slope (Right Foot)")
plt.xlabel("Time (s)")
plt.ylabel("GRF (N)")
plt.title("GRF and Slope for Force Plate 1 (Left Foot)")
plt.legend()

plt.tight_layout()
plt.show()