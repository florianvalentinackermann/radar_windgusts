import os
import pandas as pd
from pathlib import Path

# Initialize list to store valid times
valid_times = []

# Base directory path
base_path = Path("/scratch/mch/fackerma/orders/TRT_processing_3/")

# Process each year directory
for year in range(2019, 2024):
    year_dir = base_path / f"{year}"
    
    if not year_dir.exists():
        continue

    # Process each .pkl file
    for pkl_file in year_dir.glob("TRT_*.pkl"):
        filename = pkl_file.stem
        parts = filename.split("_")
        
        if len(parts) >= 3:
            try:
                date_str = parts[1]
                time_str = parts[2]
                formatted_time = f"{date_str} {time_str[:2]}:{time_str[2:]}"
                
                df = pd.read_pickle(pkl_file)
                if 'Gust_Flag' in df.columns:
                    if df['Gust_Flag'].isin(['Yes', 'No']).any():
                        valid_times.append(formatted_time)
            except Exception as e:
                print(f"Error processing {pkl_file}: {str(e)}")

# Create and format final dataframe
result_df = pd.DataFrame({'Valid_Time': valid_times})
result_df['Valid_Time'] = pd.to_datetime(result_df['Valid_Time'])
result_df = result_df.sort_values('Valid_Time').reset_index(drop=True)
result_df['Valid_Time'] = result_df['Valid_Time'].dt.strftime('%Y-%m-%d %H:%M')

# Save to specified location
output_path = base_path.parent / "Extraction_dates_3.pkl"
result_df.to_pickle(output_path)
print(f"\n✅ Successfully saved {len(result_df)} entries to:\n{output_path}")

# Optional: Verify load
test_load = pd.read_pickle(output_path)
print(f"\nVerification: Loaded {len(test_load)} records from saved file")
