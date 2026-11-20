
# TENG-Gait Analytics
# Phase 2: Signal Processing
# Raw vs Filtered Signal Comparison

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.signal import butter, sosfiltfilt

# ---------------------------------
# STEP 1: Load experimental dataset
# ---------------------------------

file_path = "data/raw/physionet/FP_S1_0.5.csv"

data = pd.read_csv(file_path)

# ---------------------------------
# STEP 2: Define sampling parameters
# ---------------------------------

fs = 1000  # Sampling frequency in Hz

cutoff_frequency = 20  # Exploratory cutoff in Hz

filter_order = 4

# ---------------------------------
# STEP 3: Extract vertical force
# ---------------------------------

raw_signal = data["1:Fz"].to_numpy()

# ---------------------------------
# STEP 4: Design Butterworth filter
# ---------------------------------

sos = butter(
    filter_order,
    cutoff_frequency,
    btype="lowpass",
    fs=fs,
    output="sos"
)

# ---------------------------------
# STEP 5: Apply zero-phase filtering
# ---------------------------------

filtered_signal = sosfiltfilt(sos, raw_signal)

# ---------------------------------
# STEP 6: Generate time axis
# ---------------------------------

time = np.arange(len(raw_signal)) / fs

# ---------------------------------
# STEP 7: Display signal statistics
# ---------------------------------

print("\nRAW SIGNAL STATISTICS")

print("Mean:", np.mean(raw_signal))
print("Standard deviation:", np.std(raw_signal))
print("Maximum:", np.max(raw_signal))
print("Minimum:", np.min(raw_signal))

print("\nFILTERED SIGNAL STATISTICS")

print("Mean:", np.mean(filtered_signal))
print("Standard deviation:", np.std(filtered_signal))
print("Maximum:", np.max(filtered_signal))
print("Minimum:", np.min(filtered_signal))

# ---------------------------------
# STEP 8: Plot first 10 seconds
# ---------------------------------

duration = 10

samples = int(duration * fs)

plt.figure(figsize=(14, 7))

plt.plot(
    time[:samples],
    raw_signal[:samples],
    label="Raw Signal",
    alpha=0.45
)

plt.plot(
    time[:samples],
    filtered_signal[:samples],
    label="Filtered Signal",
    linewidth=2
)

plt.title("Raw vs Filtered Vertical Force Signal")

plt.xlabel("Time (seconds)")
plt.ylabel("Vertical Force (dataset units)")

plt.legend()
plt.grid(True)

plt.tight_layout()

plt.show()

print("\nSIGNAL PROCESSING COMPLETE")

import numpy as np

# Calculate standard deviation
raw_std = np.std(raw_signal)
filtered_std = np.std(filtered_signal)

# Calculate RMS difference
rms_difference = np.sqrt(
    np.mean((raw_signal - filtered_signal) ** 2)
)

print("\n--- FILTER VALIDATION RESULTS ---")

print("Raw Signal Standard Deviation:", raw_std)

print("Filtered Signal Standard Deviation:", filtered_std)

print("RMS Difference:", rms_difference)