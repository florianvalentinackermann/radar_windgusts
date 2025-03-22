import pandas as pd
from pathlib import Path

valid_times = []
base_path = Path("/scratch/mch/fackerma/orders/TRT_processing_1/")
total_files = 0
files_with_gust = 0

print("Starting processing...\n")

for year in range(2019, 2024):
    year_dir = base_path / f"trt_processing_testorder_{year}"
    
    if not year_dir.exists():
        print(f"⚠️  Missing directory: {year_dir}")
        continue

    print(f"Processing {year}...")
    year_files = list(year_dir.glob("TRT_*.pkl"))
    
    if not year_files:
        print(f"   ❗ No files found in {year_dir}")
        continue

    for pkl_file in year_files:
        total_files += 1
        try:
            # Filename parsing
            filename = pkl_file.stem
            parts = filename.split("_")
            
            if len(parts) < 3:
                print(f"   ❗ Invalid filename format: {filename}")
                continue
                
            # Date/time parsing
            date_str = parts[1]
            time_str = parts[2]
            formatted_time = f"{date_str} {time_str[:2]}:{time_str[2:]}"
            
            # Data loading
            df = pd.read_pickle(pkl_file)
            
            # Column check
            if 'Gust_Flag' not in df.columns:
                print(f"   ❗ Missing Gust_Flag in {filename}")
                continue
                
            # Value check
            gust_values = df['Gust_Flag'].unique()
            has_valid = any(x in {'Yes', 'No'} for x in gust_values)
            
            if has_valid:
                valid_times.append(formatted_time)
                files_with_gust += 1
                print(f"   ✅ Valid entry: {formatted_time}")
            else:
                print(f"   ❌ No Yes/No values in {filename}")
                print(f"      Found values: {gust_values}")

        except Exception as e:
            print(f"   🛑 Error processing {filename}: {str(e)}")
            continue

# Diagnostic summary
print("\n=== Processing Summary ===")
print(f"Total directories checked: {2023-2019+1}")
print(f"Total files processed: {total_files}")
print(f"Files with Gust_Flag column: {files_with_gust}")
print(f"Valid entries found: {len(valid_times)}\n")

# Create and save dataframe if entries found
if valid_times:
    result_df = pd.DataFrame({'Valid_Time': valid_times})
    result_df['Valid_Time'] = pd.to_datetime(result_df['Valid_Time'])
    result_df = result_df.sort_values('Valid_Time').reset_index(drop=True)
    result_df['Valid_Time'] = result_df['Valid_Time'].dt.strftime('%Y-%m-%d %H:%M')
    
    output_path = base_path.parent / "Extraction_dates.pkl"
    result_df.to_pickle(output_path)
    print(f"✅ Successfully saved {len(result_df)} entries to:\n{output_path}")
else:
    print("❌ No valid entries found. Check diagnostic messages above.")

