import pandas as pd
import numpy as np


##
# Open the dataset
##

file_path = '/scratch/mch/fackerma/orders/TRT_modelsetup_3/Model_Setup_3_b.pkl'

df = pd.read_pickle(file_path)


exclude_cols = ['traj_ID', 'yyyymmddHHMM', 'Gust_Flag', 'STA_Marker', 'CS_Marker', 'ESWD_Marker', 'geometry']

for col in df.columns:
    if col not in exclude_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

df['vel_x'] = pd.to_numeric(df['vel_x'], errors='coerce')
df['vel_y'] = pd.to_numeric(df['vel_y'], errors='coerce')

df['velocity'] = np.sqrt(df['vel_x']**2 + df['vel_y']**2)

df['Z_top_smoothness'] = df['ZH_20_height'] - df['ZH_45_height'] 

df['CoreAspect_ratio'] = df['ZH_45_height'] / df['area']


##
# Random probabilistic NA filling 
##


# Columns to impute
columns_to_impute = ['IC', 'area45', 'area57', 'dBZmax']

for col in columns_to_impute:
    # Get non-NA values in the 'Yes' group for the current column
    yes_group = df[df['Gust_Flag'] == 'Yes']
    non_na_values = yes_group[col].dropna()
    
    if not non_na_values.empty:
        # Calculate probabilities for each unique value
        value_counts = non_na_values.value_counts(normalize=True)
        values = value_counts.index.tolist()
        probabilities = value_counts.values
        
        # Identify NA indices to fill
        na_mask = (df['Gust_Flag'] == 'Yes') & (df[col].isna())
        
        # Sample and fill
        df.loc[na_mask, col] = np.random.choice(
            values, 
            size=na_mask.sum(), 
            p=probabilities
        )

# Impute L_TOT with the sum of IC and CG where L_TOT is NA
na_mask = df['L_TOT'].isna()
df.loc[na_mask, 'L_TOT'] = df.loc[na_mask, 'IC'] + df.loc[na_mask, 'CG']

df.loc[df['KDP_com_height'] < 0, 'KDP_com_height'] = 0


##
# Problematic ID Investigation
##

# Only keep rows where Gust_Flag is 'Yes' or 'No'
df_events = df[df['Gust_Flag'].isin(['Yes', 'No'])].copy()

# Ensure timestamp is a datetime type
df_events['timestamp'] = pd.to_datetime(df_events['timestamp'])

# Helper function to check each traj_ID
def check_separation(group):
    # Sort by timestamp
    group = group.sort_values('timestamp')
    # Find indices where Gust_Flag changes
    flag_changes = group['Gust_Flag'] != group['Gust_Flag'].shift()
    change_indices = group.index[flag_changes].tolist()
    # For each transition, check the time difference
    for i in range(1, len(change_indices)):
        idx_prev = change_indices[i-1]
        idx_curr = change_indices[i]
        t_prev = group.loc[idx_prev, 'timestamp']
        t_curr = group.loc[idx_curr, 'timestamp']
        # If the time difference is less than 15 minutes, return False
        if (t_curr - t_prev).total_seconds() < 15*60:
            return False
    return True

# Find traj_IDs with both 'Yes' and 'No'
traj_ids_with_both = df_events.groupby('traj_ID')['Gust_Flag'].nunique()
traj_ids_with_both = traj_ids_with_both[traj_ids_with_both == 2].index

# Check each of these traj_IDs
problematic_ids = []
for traj_id in traj_ids_with_both:
    group = df_events[df_events['traj_ID'] == traj_id]
    if not check_separation(group):
        problematic_ids.append(traj_id)

filtered_df = df[~df['traj_ID'].isin(problematic_ids)]

# Get unique traj_IDs
unique_ids = filtered_df['traj_ID'].unique()


##
# Splitting Stratification
##

from sklearn.model_selection import train_test_split

# First split: train and temp (70%/30%) WITH STRATIFICATION
train_ids, temp_ids = train_test_split(
    unique_ids,
    test_size=0.3,
    stratify=filtered_df.groupby('traj_ID')['Gust_Flag'].first(),  # Stratify by storm outcome
    random_state=42
)

# Second split: test (20%) and validation (10%) WITH STRATIFICATION
test_ids, val_ids = train_test_split(
    temp_ids,
    test_size=0.333,
    stratify=filtered_df[filtered_df['traj_ID'].isin(temp_ids)].groupby('traj_ID')['Gust_Flag'].first(),
    random_state=42
)



##
# Manual split and event assimilation
##

# Suppose val_ids, train_ids, test_ids are numpy arrays
val_ids = np.array(val_ids)
train_ids = np.array(train_ids)
test_ids = np.array(test_ids)

manual_traj_ids = set(['2019061509450011.0','2021071217550028.0','2021071221500035.0','2021071305200023.0','2022063019350162','2023082417300207'])

# Convert arrays to sets
val_ids_set = set(val_ids)
train_ids_set = set(train_ids)
test_ids_set = set(test_ids)

# Add to validation set
val_ids_set = val_ids_set.union(manual_traj_ids)

# Remove from train and test sets
train_ids_set = train_ids_set - manual_traj_ids
test_ids_set = test_ids_set - manual_traj_ids

# Convert back to numpy arrays or lists if needed
val_ids = np.array(list(val_ids_set))
train_ids = np.array(list(train_ids_set))
test_ids = np.array(list(test_ids_set))


# Map IDs back to full dataframe
train_df = filtered_df[filtered_df['traj_ID'].isin(train_ids)]
test_df = filtered_df[filtered_df['traj_ID'].isin(test_ids)]
val_df = filtered_df[filtered_df['traj_ID'].isin(val_ids)]

train_df = train_df.sort_values(['traj_ID', 'yyyymmddHHMM'])
test_df = test_df.sort_values(['traj_ID', 'yyyymmddHHMM'])
val_df = val_df.sort_values(['traj_ID', 'yyyymmddHHMM'])

# Function to print stats for each split
def print_split_stats(df, name):
    total = len(df)
    unique_ids = df['traj_ID'].nunique()
    yes_count = (df['Gust_Flag'] == 'Yes').sum()
    no_count = (df['Gust_Flag'] == 'No').sum()
    ratio = yes_count / no_count if no_count > 0 else float('inf')
    print(f"{name}:")
    print(f"  Number of rows: {total}")
    print(f"  Number of unique traj_IDs: {unique_ids}")
    print(f"  'Yes' count: {yes_count}")
    print(f"  'No' count: {no_count}")
    print(f"  'Yes':'No' ratio: {yes_count}:{no_count} ({ratio:.3f})")
    print("-" * 40)

# Only keep rows with 'Yes' or 'No'
filtered_flags = filtered_df[filtered_df['Gust_Flag'].isin(['Yes', 'No'])]
# Count 'Yes' and 'No' per traj_ID
count_df = (
    filtered_flags
    .groupby('traj_ID')['Gust_Flag']
    .value_counts()
    .unstack(fill_value=0)
    .reset_index()
)



##
# Define features
##

features_no_lags = [#'area', 
                    #'RANKr', 
                    'dBZmax', 
                    'velocity', 
                    #'VIL', 
                    #'Age', 
                    'area45',
                    #'area57',
                    #'POH', 
                    #'MESHS',
                    'CG', 
                    'IC', 
                    'L_TOT',
                    'rad_shear_max', 
                    'rad_shear_max_height', 
                    #'rad_shear_95th_percentile', 
                    #'rad_shear_95th_percentile_height',
                    'rad_shear_percent_above_2.5', 
                    'rad_shear_percent_above_2', 
                    'rad_shear_percent_above_1.5', 
                    'rad_shear_percent_above_1', 
                    'rad_shear_percent_above_0.5',
                    'ZH_com_height', 
                    #'ZH_max', 
                    #'ZH_max_height', 
                    #'ZH_95th_percentile', 
                    #'ZH_95th_percentile_height', 
                    #'ZH_percent_above_55', 
                    #'ZH_percent_above_50', 
                    'ZH_percent_above_45', 
                    'ZH_percent_above_40', 
                    'ZH_percent_above_35', 
                    'ZH_percent_above_30',
                    'ZH_45_height', 
                    #'ZH_20_height', 
                    'Z_top_smoothness', 
                    #'CoreAspect_ratio', 
                    #'KDP_com_height', 
                    #'KDP_max', 
                    #'KDP_max_height', 
                    'KDP_95th_percentile', 
                    #'KDP_95th_percentile_height',
                    'KDP_percent_above_2', 
                    'KDP_percent_above_1.5', 
                    'KDP_percent_above_1', 
                    'KDP_percent_above_0.5'
                    ]   


all_features = features_no_lags 

train_df_events = train_df[train_df['Gust_Flag'].isin(['Yes', 'No'])].copy()
val_df_events = val_df[val_df['Gust_Flag'].isin(['Yes', 'No'])].copy()
test_df_events = test_df[test_df['Gust_Flag'].isin(['Yes', 'No'])].copy()

# 5. Drop NaNs only for required features (MODIFIED)
val_df_events = val_df_events.dropna(how='all', subset=all_features)
train_df_events = train_df_events.dropna(how='all', subset=all_features)
test_df_events = test_df_events.dropna(how='all', subset=all_features)





##
# EFS
##

from mlxtend.feature_selection import ExhaustiveFeatureSelector as EFS
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

X_train = train_df_events[all_features]
y_train = train_df_events['Gust_Flag']
X_val = val_df_events[all_features]
y_val = val_df_events['Gust_Flag']
X_test = test_df_events[all_features]
y_test = test_df_events['Gust_Flag']

# Define a custom scoring function for CSI
def csi_scorer(estimator, X, y):
    y_pred = estimator.predict(X)
    cm = confusion_matrix(y, y_pred, labels=['Yes', 'No'])
    a = cm[0, 0]  # True Positives
    b = cm[1, 0]  # False Positives
    c = cm[0, 1]  # False Negatives
    csi = a / (a + b + c) if (a + b + c) > 0 else 0
    return csi

# Set the range for number of features to consider
min_features = 24
max_features = 24

# Initialize the Random Forest model
rf = RandomForestClassifier(class_weight='balanced', n_jobs=-1, random_state=42)

# Run exhaustive feature selection
efs = EFS(
    rf,
    min_features=min_features,
    max_features=max_features,
    scoring=csi_scorer,
    print_progress=True,
    cv=2  # Use cv=2 for speed; increase for more robust results
)

efs = efs.fit(X_train, y_train)

# Get the best feature subset(s) and their CSI scores
best_idx = np.argmax(efs.get_metric_dict().values())
best_features = list(efs.get_metric_dict().values())[best_idx]['feature_names']
best_score = list(efs.get_metric_dict().values())[best_idx]['avg_score']

print("Best feature subset:", best_features)
print("Best average CSI:", best_score)

# To see all combinations ranked by CSI:
results = sorted(
    [(v['feature_names'], v['avg_score']) for v in efs.get_metric_dict().values()],
    key=lambda x: x[1], reverse=True
)
for features, score in results:
    print(f"Features: {features}, CSI: {score:.3f}")