import pandas as pd
from sklearn.datasets import fetch_california_housing
import numpy as np

# Load the dataset
housing = fetch_california_housing(as_frame=True)

# Combine features and target into one DataFrame
df = housing.frame  # This already combines features and target

# Save to CSV file
df.to_csv('california_housing.csv', index=False)

print("Dataset has been saved to 'california_housing.csv'")
print(f"Dataset shape: {df.shape}")
print("\nColumns in the dataset:")
for col in df.columns:
    print(f"- {col}")
