# Import the data
import os
import pandas as pd
import numpy as np

# Define the base directory and file names
base_dir = "/scratch/mch/fackerma/orders/TRT_processing_2/"
yearly_files = [
    "TRT_2019_05-10.pkl",
    "TRT_2020_05-10.pkl",
    "TRT_2021_05-10.pkl",
    "TRT_2022_05-10.pkl",
    "TRT_2023_05-10.pkl",
]

# Load and merge dataframes
dfs = []
for file_name in yearly_files:
    file_path = os.path.join(base_dir, file_name)
    if os.path.exists(file_path):
        print(f"Loading {file_name}...")
        df = pd.read_pickle(file_path)
        dfs.append(df)
    else:
        print(f"⚠️ File not found: {file_name}")

if not dfs:
    print("No data loaded. Exiting.")
    exit()

merged_df = pd.concat(dfs, ignore_index=True)

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Set the default font size for all text elements
plt.rcParams.update({'font.size': 30})

fig, ax = plt.subplots(figsize=(24, 16))

merged_df['vel_x'] = pd.to_numeric(merged_df['vel_x'], errors='coerce')
merged_df['vel_y'] = pd.to_numeric(merged_df['vel_y'], errors='coerce')

clean_df = merged_df.dropna(subset=['vel_x', 'vel_y', 'Gust_Flag'])

# Plot 'Yes' gusts
sns.kdeplot(
    data=clean_df[clean_df['Gust_Flag'] == 'Yes'],
    x='vel_x',
    y='vel_y',
    cmap='Reds',
    thresh=0.1,
    ax=ax
)

# Plot 'No' gusts
sns.kdeplot(
    data=clean_df[clean_df['Gust_Flag'] == 'No'],
    x='vel_x',
    y='vel_y',
    cmap='Blues',
    thresh=0.1,
    ax=ax
)

# Plot '-' gusts
sns.kdeplot(
    data=clean_df[clean_df['Gust_Flag'] == '-'],
    x='vel_x',
    y='vel_y',
    cmap='Greens',
    thresh=0.1,
    ax=ax
)

# Add vector field elements
ax.quiver(0, 0, 1, 0, color='k', scale_units='xy', scale=1)  # East arrow
ax.quiver(0, 0, 0, 1, color='k', scale_units='xy', scale=1)  # North arrow

# Set labels and limits with fontsize 20
ax.set_xlabel('Eastward Velocity (km h⁻¹)', fontsize=30)
ax.set_ylabel('Northward Velocity (km h⁻¹)', fontsize=30)
ax.set_xlim(-30, 90)
ax.set_ylim(-35, 80)

# Increase tick label font size
ax.tick_params(axis='both', which='major', labelsize=30)

# Create custom legend with fontsize 20
red_patch = mpatches.Patch(color='red', label='5min Gust Cells')
blue_patch = mpatches.Patch(color='blue', label='5min Non-Gust Cells')
black_patch = mpatches.Patch(color='black', label='5min Uncertain Cells')
ax.legend(handles=[red_patch, blue_patch, black_patch], loc='upper left', fontsize=30)

plt.savefig("/users/fackerma/newproject1/figures/Ground_Truth/Speed_Densityplots2.png")