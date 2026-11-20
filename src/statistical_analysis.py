
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path

# -----------------------------------------
# 1. LOAD FEATURE DATASET
# -----------------------------------------

file_path = Path(
    "data/processed/physionet/gait_features_S1_0.5.csv"
)

df = pd.read_csv(file_path)

print("Dataset loaded successfully!")

# Exclude the potentially incomplete first contact
df = df[
    df["Recording Boundary Contact"].astype(str).str.lower()
    == "false"
].copy()

# The first remaining contact interval is measured
# from the recording-boundary contact, so exclude it.
df.loc[df.index[0], "Interval Since Previous Contact (s)"] = np.nan

print("Number of complete contact intervals:", len(df))


# -----------------------------------------
# 2. SELECT FEATURES
# -----------------------------------------

features = [
    "Stance Duration (s)",
    "Peak Force (N)",
    "Mean Force (N)",
    "Force Std (N)",
    "Force Impulse (N.s)",
    "Interval Since Previous Contact (s)"
]

analysis_data = df[features]


# -----------------------------------------
# 3. DESCRIPTIVE STATISTICS
# -----------------------------------------

print("\n--- DESCRIPTIVE STATISTICS ---")

statistics = analysis_data.describe().T

print(statistics)

output_folder = Path("data/processed/physionet")
output_folder.mkdir(parents=True, exist_ok=True)

statistics.to_csv(
    output_folder / "descriptive_statistics.csv"
)


# -----------------------------------------
# 4. OUTLIER DETECTION USING IQR
# -----------------------------------------

print("\n--- OUTLIER DETECTION ---")

outlier_results = []

for feature in features:

    values = analysis_data[feature].dropna()

    Q1 = values.quantile(0.25)
    Q3 = values.quantile(0.75)

    IQR = Q3 - Q1

    lower_limit = Q1 - 1.5 * IQR
    upper_limit = Q3 + 1.5 * IQR

    outliers = values[
        (values < lower_limit) |
        (values > upper_limit)
    ]

    outlier_results.append({
        "Feature": feature,
        "Lower Limit": lower_limit,
        "Upper Limit": upper_limit,
        "Outlier Count": len(outliers)
    })

    print("\nFeature:", feature)
    print("Lower limit:", lower_limit)
    print("Upper limit:", upper_limit)
    print("Number of outliers:", len(outliers))


outlier_df = pd.DataFrame(outlier_results)

outlier_df.to_csv(
    output_folder / "outlier_analysis.csv",
    index=False
)


# -----------------------------------------
# 5. CORRELATION ANALYSIS
# -----------------------------------------

print("\n--- CORRELATION MATRIX ---")

correlation_matrix = analysis_data.corr()

print(correlation_matrix.round(3))

correlation_matrix.to_csv(
    output_folder / "correlation_matrix.csv"
)


# -----------------------------------------
# 6. CORRELATION HEATMAP
# -----------------------------------------

plt.figure(figsize=(11, 8))

sns.heatmap(
    correlation_matrix,
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    linewidths=0.5
)

plt.title("Gait Feature Correlation Heatmap")

plt.tight_layout()

plt.savefig(
    output_folder / "correlation_heatmap.png",
    dpi=300
)

plt.show()


# -----------------------------------------
# 7. SAVE FINAL REPORT
# -----------------------------------------

print("\n--- ANALYSIS COMPLETED ---")

print("Generated files:")

print("1. descriptive_statistics.csv")
print("2. outlier_analysis.csv")
print("3. correlation_matrix.csv")
print("4. correlation_heatmap.png")

print("\nAll statistical analysis files saved successfully!")
