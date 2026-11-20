
# TENG-Gait Analytics
# Phase 1: Force vs Time Visualization

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. Load experimental dataset
file_path = "data/raw/physionet/FP_S1_0.5.csv"

data = pd.read_csv(file_path)

# 2. Define sampling frequency
sampling_frequency = 1000  # Hz

# 3. Generate time axis
sample_number = np.arange(len(data))

time = sample_number / sampling_frequency

# 4. Extract vertical force measurements
force_plate_1 = data["1:Fz"]
force_plate_2 = data["2:Fz"]

# 5. Print basic information
print("Number of observations:", len(data))
print("Sampling frequency:", sampling_frequency, "Hz")
print("Recording duration:", time[-1], "seconds")

# 6. Plot both force signals
plt.figure(figsize=(14, 6))

plt.plot(time, force_plate_1,
         label="Force Plate 1")

plt.plot(time, force_plate_2,
         label="Force Plate 2",
         alpha=0.7)

# 7. Graph formatting
plt.title("Vertical Ground Reaction Force vs Time")

plt.xlabel("Time (seconds)")
plt.ylabel("Vertical Force (dataset units)")

plt.legend()
plt.grid(True)

plt.tight_layout()

# 8. Display graph
plt.show()
