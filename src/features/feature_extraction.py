
import pandas as pd
import numpy as np

from scipy.signal import butter, sosfiltfilt
from pathlib import Path

# -----------------------------------
# 1. LOAD DATA
# -----------------------------------

data_path = Path("data/raw/physionet/FP_S1_0.5.csv")

data = pd.read_csv(data_path)

raw_signal = data["1:Fz"].to_numpy(dtype=float)

fs = 1000

time = np.arange(len(raw_signal)) / fs

print("Dataset loaded successfully!")


# -----------------------------------
# 2. FILTER SIGNAL
# -----------------------------------

sos = butter(
    4,
    20,
    btype="lowpass",
    fs=fs,
    output="sos"
)

filtered_signal = sosfiltfilt(sos, raw_signal)


# -----------------------------------
# 3. DETECT CONTACT INTERVALS
# -----------------------------------

threshold = 20

contact = filtered_signal > threshold

edges = np.diff(
    np.concatenate(([False], contact, [False])).astype(int)
)

starts = np.where(edges == 1)[0]
ends = np.where(edges == -1)[0]

minimum_stance_samples = int(0.10 * fs)

features = []

for start, end in zip(starts, ends):

    if end - start < minimum_stance_samples:
        continue

    segment = filtered_signal[start:end]

    stance_duration = (end - start) / fs

    peak_force = np.max(segment)

    mean_force = np.mean(segment)

    force_std = np.std(segment)

    force_impulse = np.sum(segment) / fs

    features.append({
        "Contact Start (s)": start / fs,
        "Contact End (s)": end / fs,
        "Stance Duration (s)": stance_duration,
        "Peak Force (N)": peak_force,
        "Mean Force (N)": mean_force,
        "Force Std (N)": force_std,
        "Force Impulse (N.s)": force_impulse
    })


# -----------------------------------
# 4. CREATE FEATURE TABLE
# -----------------------------------

features_df = pd.DataFrame(features)

if len(features_df) > 0:

    features_df["Interval Since Previous Contact (s)"] = (
        features_df["Contact Start (s)"].diff()
    )

    # First contact may be incomplete because
    # recording starts while force is already present.
    features_df["Recording Boundary Contact"] = False
    features_df.loc[0, "Recording Boundary Contact"] = True

    print("\n--- GAIT FEATURE EXTRACTION RESULTS ---")

    print(features_df.to_string(index=False))

    # Exclude the potentially incomplete first contact
    # from the summary statistics.
    complete_contacts = features_df.iloc[1:]

    print("\n--- SUMMARY OF COMPLETE CONTACTS ---")

    print(
        "Average stance duration:",
        complete_contacts["Stance Duration (s)"].mean()
    )

    print(
        "Average peak force:",
        complete_contacts["Peak Force (N)"].mean()
    )

    print(
        "Average mean force:",
        complete_contacts["Mean Force (N)"].mean()
    )

    print(
        "Average force impulse:",
        complete_contacts["Force Impulse (N.s)"].mean()
    )

else:

    print("No contact intervals detected.")


# -----------------------------------
# 5. SAVE FEATURE DATASET
# -----------------------------------

output_folder = Path("data/processed/physionet")

output_folder.mkdir(parents=True, exist_ok=True)

output_file = output_folder / "gait_features_S1_0.5.csv"

features_df.to_csv(output_file, index=False)

print("\nFeature dataset saved successfully!")
print("File location:", output_file)
