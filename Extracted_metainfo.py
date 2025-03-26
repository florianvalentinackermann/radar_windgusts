import os
import pandas as pd
import glob

# Define the main directory
main_dir = '/scratch/mch/fackerma/orders/TRT_processing_2/'
output_dir = '/scratch/mch/fackerma/orders/'

# Initialize an empty list to store all relevant DataFrames
all_dfs = []

# Iterate through all subdirectories
for subdir in os.listdir(main_dir):
    subdir_path = os.path.join(main_dir, subdir)
    if os.path.isdir(subdir_path):
        # Find all .pkl files in the subdirectory
        pkl_files = glob.glob(os.path.join(subdir_path, '*.pkl'))
        
        for file in pkl_files:
            try:
                # Read the .pkl file
                df = pd.read_pickle(file)
                
                # Check if the DataFrame is not empty and has 'Gust_Flag' column
                if not df.empty and 'Gust_Flag' in df.columns:
                    # Filter rows where Gust_Flag is 'Yes' or 'No'
                    df_filtered = df[df['Gust_Flag'].isin(['Yes', 'No'])]
                    
                    if not df_filtered.empty:
                        all_dfs.append(df_filtered)
            except Exception as e:
                print(f"Error processing file {file}: {str(e)}")

# Combine all DataFrames, keeping all columns and filling missing values with NA
if all_dfs:
    final_df = pd.concat(all_dfs, ignore_index=True, sort=False)
    
    # Save the final DataFrame as 'Extracted_metainfo.pkl'
    output_path = os.path.join(output_dir, 'Extracted_metainfo.pkl')
    final_df.to_pickle(output_path)
    print(f"DataFrame saved as {output_path}")
    
    # Display information about the final DataFrame
    print(final_df.info())
    print(final_df.head())
else:
    print("No data found matching the criteria. No file was saved.")
