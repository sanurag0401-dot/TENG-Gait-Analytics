
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Load extracted features
file_path = Path(
    "data/processed/physionet/gait_features_S1_0.5.csv"
)

df = pd.read_csv(file_path)

# Exclude the potentially incomplete first contact
df = df[df["Recording Boundary Contact"] == False]

# Features to visualize
features = [
    "Stance Duration (s)",
    "Peak Force (N)",
    "Mean Force (N)",
    "Force Impulse (N.s)"
]

# Create four individual histograms
for feature in features:

    plt.figure(figsize=(9, 5))

    plt.hist(
        df[feature].dropna(),
        bins=12,
        edgecolor="black"
    )

    plt.title(f"Distribution of {feature}")
    plt.xlabel(feature)
    plt.ylabel("Number of Contacts")

    plt.grid(alpha=0.3)
    plt.tight_layout()

    filename = feature.replace(" ", "_").replace("/", "_")

    plt.savefig(
        f"data/processed/physionet/{filename}.png",
        dpi=300
    )

    plt.show()

print("All feature distribution plots generated!")
