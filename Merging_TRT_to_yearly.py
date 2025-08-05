import os
import pandas as pd
from datetime import datetime
import warnings

base_dir = "/scratch/mch/fackerma/orders/TRT_processing_3/"
output_template = "TRT_{year}_05-10_new.pkl"

def process_year(year):
    year_dir = os.path.join(base_dir, str(year))
    if not os.path.exists(year_dir):
        print(f"⚠️ Directory not found: {year_dir}")
        return

    try:
        # Collect and validate files
        files = [f for f in os.listdir(year_dir) 
                if f.startswith(f'TRT_{year}') and f.endswith('.pkl')]
        
        if not files:
            print(f"⏩ No files found for {year}")
            return

        # Sort files chronologically
        files.sort(key=lambda x: datetime.strptime(x[4:19], "%Y-%m-%d_%H%M"))
        
        # Process data
        dfs = []
        for file in files:
            try:
                file_path = os.path.join(year_dir, file)
                df = pd.read_pickle(file_path)
                timestamp = datetime.strptime(file[4:19], "%Y-%m-%d_%H%M")
                df.insert(0, 'timestamp', timestamp)  # Add as first column
                dfs.append(df)
            except Exception as e:
                print(f"❌ Error processing {file}: {str(e)}")
                continue

        if not dfs:
            print(f"⛔ No valid data for {year}")
            return

        # Merge and save
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            merged_df = pd.concat(dfs, axis=0, ignore_index=True, join='outer')
            merged_df.sort_values('timestamp', inplace=True)

        output_path = os.path.join(base_dir, output_template.format(year=year))
        merged_df.to_pickle(output_path, protocol=5)
        print(f"✅ Successfully processed {year} ({len(dfs)} files)")
        
    except Exception as e:
        print(f"🔥 Critical error in {year}: {str(e)}")

# Process all years with error protection
for year in range(2019, 2024):
    process_year(year)

print("✨ Processing complete! Verify output files.")
