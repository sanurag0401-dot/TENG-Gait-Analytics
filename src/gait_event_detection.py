
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.signal import butter, sosfiltfilt
from pathlib import Path

# -----------------------------------
# 1. LOAD THE DATASET
# -----------------------------------

data_path = Path("data/raw/physionet/FP_S1_0.5.csv")

data = pd.read_csv(data_path)

# Extract vertical force from Force Plate 1
raw_signal = data["1:Fz"].to_numpy(dtype=float)

# Sampling frequency
fs = 1000

# Generate time axis
time = np.arange(len(raw_signal)) / fs

print("Dataset loaded successfully!")
print("Number of samples:", len(raw_signal))


# -----------------------------------
# 2. FILTER THE SIGNAL
# -----------------------------------

cutoff_frequency = 20
filter_order = 4

sos = butter(
    filter_order,
    cutoff_frequency,
    btype="lowpass",
    fs=fs,
    output="sos"
)

filtered_signal = sosfiltfilt(sos, raw_signal)


# -----------------------------------
# 3. DETECT FOOT CONTACT
# -----------------------------------

# Exploratory force threshold in Newtons
threshold = 20

contact = filtered_signal > threshold

# Identify where contact begins and ends
edges = np.diff(
    np.concatenate(([False], contact, [False])).astype(int)
)

starts = np.where(edges == 1)[0]
ends = np.where(edges == -1)[0]

# Remove very short contact intervals
minimum_stance_samples = int(0.10 * fs)

events = []

for start, end in zip(starts, ends):

    duration_samples = end - start

    if duration_samples >= minimum_stance_samples:

        heel_strike_time = start / fs
        toe_off_time = end / fs

        stance_duration = (
            toe_off_time - heel_strike_time
        )

        events.append({
            "Heel Strike (s)": heel_strike_time,
            "Toe-Off (s)": toe_off_time,
            "Stance Duration (s)": stance_duration
        })


# -----------------------------------
# 4. DISPLAY GAIT EVENTS
# -----------------------------------

events_df = pd.DataFrame(events)

print("\n--- GAIT EVENT DETECTION RESULTS ---")

if events_df.empty:

    print("No gait events detected.")
    print("Check the force threshold and signal.")

else:

    print(events_df.to_string(index=False))

    print("\nTotal detected stance phases:", len(events_df))

    # Calculate intervals between consecutive heel strikes
    if len(events_df) >= 2:

        heel_strikes = events_df["Heel Strike (s)"].to_numpy()

        gait_cycle_intervals = np.diff(heel_strikes)

        print(
            "\nMean interval between consecutive contacts:",
            np.mean(gait_cycle_intervals),
            "seconds"
        )


# -----------------------------------
# 5. SAVE RESULTS
# -----------------------------------

output_folder = Path("data/processed/physionet")
output_folder.mkdir(parents=True, exist_ok=True)

output_file = output_folder / "FP_S1_0.5_gait_events.csv"

events_df.to_csv(output_file, index=False)

print("\nResults saved to:", output_file)


# -----------------------------------
# 6. PLOT DETECTED GAIT EVENTS
# -----------------------------------


plt.figure(figsize=(14, 6))

plt.plot(
    time,
    filtered_signal,
    label="Filtered Vertical Force",
    color="blue"
)

plt.axhline(
    y=threshold,
    color="red",
    linestyle="--",
    label="Contact Threshold"
)

for i, event in events_df.iterrows():

    plt.axvline(
        event["Heel Strike (s)"],
        color="green",
        linestyle=":",
        alpha=0.7,
        label="Contact Start" if i == 0 else None
    )

    plt.axvline(
        event["Toe-Off (s)"],
        color="orange",
        linestyle=":",
        alpha=0.7,
        label="Contact End" if i == 0 else None
    )

plt.xlabel("Time (seconds)")
plt.ylabel("Vertical Force (N)")

plt.title("Gait Event Detection - Force Plate 1")

# Correct axis limits


plt.ylim(
    min(0, np.min(filtered_signal) - 50),
    np.max(filtered_signal) + 100
)

plt.legend()
plt.grid(True)
plt.tight_layout()

plt.xlim(0, time[-1])
plt.savefig(
    "data/processed/physionet/gait_event_detection.png",
    dpi=300
)

plt.show()
