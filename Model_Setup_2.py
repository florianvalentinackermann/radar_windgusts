import os
from datetime import datetime
import numpy as np
from shapely.geometry import Point
from scipy.ndimage import center_of_mass
import pandas as pd
import matplotlib.pyplot as plt
from pyproj import Transformer
from shapely.prepared import prep





# Define the Swiss grid (adjusted to match data dimensions)
chx = np.arange(255000, 255000 + 710 * 1000, 1000)  # Easting values (710 points)
chy = sorted(np.arange(-160000, -160000 + 640 * 1000, 1000), reverse=True)  # Northing values (640 points)
X, Y = np.meshgrid(chx, chy)

# Initialize transformer for Swiss grid to WGS84 (EPSG:21781 to EPSG:4326 PlateCarree)
transformer = Transformer.from_crs(21781, 4326, always_xy=True)
clons, clats = transformer.transform(X, Y)



def calculate_metrics(filtered_df, clons, clats, ZH, rad_shear, KDP):
    # Precompute grid properties
    ny, nx = clons.shape
    nz = ZH.shape[2]
    clons_flat = clons.ravel()
    clats_flat = clats.ravel()
    
    # Initialize results storage
    results = {
        'ZH_com_height': [], 'ZH_percent_above_45': [], 'ZH_percent_above_50': [], 'ZH_percent_above_55': [],
        'KDP_com_height': [], 'KDP_percent_above_2': [], 'KDP_percent_above_1.5': [], 'KDP_percent_above_1': [],
        'rad_shear_max': [], 'rad_shear_percent_above_2.5': [], 'rad_shear_percent_above_2': [], 'rad_shear_percent_above_1.5': [],
        'area_p': []   # New column for polygon area
    }

    for _, row in filtered_df.iterrows():
        poly = row['geometry']
        
        # Calculate polygon area in square meters
        poly_area = poly.area  # Shapely's area property calculates area in native CRS units (meters for Swiss grid)
        results['area_p'].append(poly_area)
        
        prep_poly = prep(poly)
        minx, miny, maxx, maxy = poly.bounds
        
        # Bounding box optimization
        bbox_mask = (clons_flat >= minx) & (clons_flat <= maxx) & \
                    (clats_flat >= miny) & (clats_flat <= maxy)
        if not bbox_mask.any():
            results = _append_defaults(results)
            continue
            
        # Vectorized containment check
        contained = np.array([prep_poly.contains(Point(p)) 
                               for p in zip(clons_flat[bbox_mask], clats_flat[bbox_mask])])
        if not contained.any():
            results = _append_defaults(results)
            continue
            
        # Create 2D mask
        mask_2d = np.zeros((ny, nx), bool)
        yx_indices = np.unravel_index(np.where(bbox_mask)[0][contained], (ny, nx))
        mask_2d[yx_indices] = True
        
        # Extend the mask to 3D by repeating along the vertical dimension
        mask_3d = np.broadcast_to(mask_2d[..., None], (ny, nx, nz))
        
        # Extract values within the polygon for ZH, KDP, rad_shear
        ZH_masked = np.where(mask_3d, ZH, np.nan)
        KDP_masked = np.where(mask_3d, KDP, np.nan)
        rad_shear_masked = np.where(mask_3d, rad_shear, np.nan)

        # Replace NaN values with 0 for calculations
        ZH_masked[np.isnan(ZH_masked)] = 0
        KDP_masked[np.isnan(KDP_masked)] = 0
        rad_shear_masked[np.isnan(rad_shear_masked)] = 0
        
        # Calculate metrics for ZH
        if np.any(ZH_masked > 0):  # Check if there are valid values
            com_ZH = center_of_mass(ZH_masked)  # Center of mass height
            results['ZH_com_height'].append(com_ZH[2])  # Use the vertical dimension (z-axis)
            results['ZH_percent_above_45'].append(np.sum(ZH_masked > 45) / np.size(ZH_masked) * 100)
            results['ZH_percent_above_50'].append(np.sum(ZH_masked > 50) / np.size(ZH_masked) * 100)
            results['ZH_percent_above_55'].append(np.sum(ZH_masked > 55) / np.size(ZH_masked) * 100)
        else:
            results['ZH_com_height'].append(np.nan)
            results['ZH_percent_above_45'].append(0)
            results['ZH_percent_above_50'].append(0)
            results['ZH_percent_above_55'].append(0)
        
        # Calculate metrics for KDP
        if np.any(KDP_masked > 0):  # Check if there are valid values
            com_KDP = center_of_mass(KDP_masked)  # Center of mass height
            results['KDP_com_height'].append(com_KDP[2])  # Use the vertical dimension (z-axis)
            results['KDP_percent_above_2'].append(np.sum(KDP_masked > 2) / np.size(KDP_masked) * 100)
            results['KDP_percent_above_1.5'].append(np.sum(KDP_masked > 1.5) / np.size(KDP_masked) * 100)
            results['KDP_percent_above_1'].append(np.sum(KDP_masked > 1) / np.size(KDP_masked) * 100)
        else:
            results['KDP_com_height'].append(np.nan)
            results['KDP_percent_above_2'].append(0)
            results['KDP_percent_above_1.5'].append(0)
            results['KDP_percent_above_1'].append(0)
        
        # Calculate metrics for rad_shear
        if np.any(rad_shear_masked > 0):  # Check if there are valid values
            results['rad_shear_max'].append(np.nanmax(rad_shear_masked))
            results['rad_shear_percent_above_2.5'].append(np.sum(rad_shear_masked > 2.5) / np.size(rad_shear_masked) * 100)
            results['rad_shear_percent_above_2'].append(np.sum(rad_shear_masked > 2) / np.size(rad_shear_masked) * 100)
            results['rad_shear_percent_above_1.5'].append(np.sum(rad_shear_masked > 1.5) / np.size(rad_shear_masked) * 100)
        else:
            results['rad_shear_max'].append(np.nan)
            results['rad_shear_percent_above_2.5'].append(0)
            results['rad_shear_percent_above_2'].append(0)
            results['rad_shear_percent_above_1.5'].append(0)

    # Add results to DataFrame
    for col in results:
        filtered_df[col] = results[col]
        
    return filtered_df

def _append_defaults(results):
    """Append default values to result lists when no valid data is found."""
    for k in results:
        default = 0 if 'percent' in k or 'max' in k else np.nan
        results[k].append(default)
    return results 



import os
import pandas as pd
import numpy as np
from datetime import datetime

# 1. Load Extraction Dates
extraction_file = "/scratch/mch/fackerma/orders/Extraction_dates_20250325130000.txt"

# Read the file into a DataFrame
extraction_dates = pd.read_csv(extraction_file)

# Parse 'Valid_Time' column into datetime format
extraction_dates['Valid_Time'] = pd.to_datetime(extraction_dates['Valid_Time'], format='%Y%m%d%H%M%S')

# Extract valid times as Python datetime objects
#valid_times = extraction_dates['Valid_Time'].dt.to_pydatetime()

# Convert valid_times to timezone-aware datetime objects (UTC)
valid_times = pd.to_datetime(extraction_dates['Valid_Time'], format='%Y%m%d%H%M%S').dt.tz_localize('UTC')



# 2. Load Merged DataFrame
base_dir = "/scratch/mch/fackerma/orders/TRT_processing_2/"
yearly_files = [
    "TRT_2019_05-10.pkl",
    "TRT_2020_05-10.pkl",
    "TRT_2021_05-10.pkl",
    "TRT_2022_05-10.pkl",
    "TRT_2023_05-10.pkl",
]

dfs = []
for file_name in yearly_files:
    file_path = os.path.join(base_dir, file_name)
    if os.path.exists(file_path):
        print(f"Loading {file_name}...")
        dfs.append(pd.read_pickle(file_path))

merged_df = pd.concat(dfs, ignore_index=True)

# 3. Filter by Valid Times and Gust Flags
# Convert merged_df timestamp to match extraction format
merged_df['timestamp'] = pd.to_datetime(merged_df['timestamp'], utc=True)


# Time filtering
time_filter = merged_df['timestamp'].isin(valid_times)
filtered_by_time = merged_df[time_filter].copy()
print(f"Number of rows after time filtering: {filtered_by_time.shape[0]}")


# Find traj_IDs with at least one Yes/No in Gust_Flag
valid_traj_ids = merged_df[merged_df['Gust_Flag'].isin(['Yes', 'No'])]['traj_ID'].unique()
traj_filter = filtered_by_time['traj_ID'].isin(valid_traj_ids)


final_df = filtered_by_time[traj_filter].copy()
print(f"Number of rows after traj_ID filtering: {final_df.shape[0]}")

# 4. Process Data with calculate_metrics
npz_base = '/scratch/mch/maregger/hailclass/convective_wind/full_composite_npz/'
output_path = "/scratch/mch/fackerma/orders/TRT_modelsetup_2/Model_Setup_1.pkl"

# Group by timestamp for NPZ loading
grouped = final_df.groupby('timestamp')

all_results = []
for timestamp, group in grouped:
    try:
        # Load corresponding NPZ file
        npz_time = timestamp.strftime('%Y%m%d%H%M00')
        npz_path = f"{npz_base}{npz_time}_conv_wind_composite_data.npz"
        
        if not os.path.exists(npz_path):
            print(f"⚠️ NPZ file not found: {npz_path}")
            continue
            
        with np.load(npz_path) as data:
            ZH = data['ZH_max']
            rad_shear = data['RAD_SHEAR_LLSD_max']
            KDP = data['KDP_max']
            
        # Process the group
        processed_group = calculate_metrics(
            filtered_df=group,
            clons=clons,
            clats=clats,
            ZH=ZH,
            rad_shear=rad_shear,
            KDP=KDP
        )
        
        all_results.append(processed_group)
        
    except Exception as e:
        print(f"Error processing {timestamp}: {str(e)}")

# 5. Save Final Output
if all_results:
    final_output = pd.concat(all_results, ignore_index=True)
    final_output.to_pickle(output_path)
    print(f"Saved final output to {output_path}")
else:
    print("No data processed - output file not created")
