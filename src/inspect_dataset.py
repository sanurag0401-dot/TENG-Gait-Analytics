
# TENG-Gait Analytics
# Phase 1: Dataset Exploration

import pandas as pd
import os

# Step 1: Define dataset location
file_path = "data/raw/physionet/FP_S1_0.5.csv"

# Step 2: Check if the dataset exists
if not os.path.exists(file_path):

    print("ERROR: Dataset not found!")
    print("Please check the file location.")

else:

    # Step 3: Load experimental data
    data = pd.read_csv(file_path)

    print("\n========== DATASET LOADED ==========")

    # Step 4: Dataset dimensions
    print("\nDataset Shape:")
    print(data.shape)

    # Step 5: Column names
    print("\nColumn Names:")
    print(data.columns.tolist())

    # Step 6: First five observations
    print("\nFirst Five Rows:")
    print(data.head())

    # Step 7: Information about dataset
    print("\nDataset Information:")
    data.info()

    # Step 8: Statistical summary
    print("\nStatistical Summary:")
    print(data.describe())

    # Step 9: Check missing values
    print("\nMissing Values:")
    print(data.isnull().sum())

    print("\n========== EXPLORATION COMPLETE ==========")
