

import os
from datetime import datetime
import numpy as np
from shapely.geometry import Point
from scipy.ndimage import center_of_mass
import pandas as pd
import matplotlib.pyplot as plt
from pyproj import Transformer
from shapely.prepared import prep



# Define date range (start and end dates in 'YYYY-MM-DD_hhmm' format)
start_date_str = '2021-07-12_2000'
end_date_str = '2021-07-13_0800'


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
        'ZH_com_height': [], 'ZH_percent_above_45': [],
        'KDP_com_height': [], 'KDP_percent_above_2': [],
        'rad_shear_max': [], 'rad_shear_percent_above_2': []
    }

    for _, row in filtered_df.iterrows():
        poly = row['geometry']
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
        else:
            results['ZH_com_height'].append(np.nan)
            results['ZH_percent_above_45'].append(0)
        
        # Calculate metrics for KDP
        if np.any(KDP_masked > 0):  # Check if there are valid values
            com_KDP = center_of_mass(KDP_masked)  # Center of mass height
            results['KDP_com_height'].append(com_KDP[2])  # Use the vertical dimension (z-axis)
            results['KDP_percent_above_2'].append(np.sum(KDP_masked > 2) / np.size(KDP_masked) * 100)
        else:
            results['KDP_com_height'].append(np.nan)
            results['KDP_percent_above_2'].append(0)
        
        # Calculate metrics for rad_shear
        if np.any(rad_shear_masked > 0):  # Check if there are valid values
            results['rad_shear_max'].append(np.nanmax(rad_shear_masked))
            results['rad_shear_percent_above_2'].append(np.sum(rad_shear_masked > 2) / np.size(rad_shear_masked) * 100)
        else:
            results['rad_shear_max'].append(np.nan)
            results['rad_shear_percent_above_2'].append(0)

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





# Define paths
input_dir = '/scratch/mch/fackerma/orders/TRT_processing_2/2021/'
output_dir = '/scratch/mch/fackerma/orders/TRT_modelsetup_2/2021/'
npz_base = '/scratch/mch/maregger/hailclass/convective_wind/full_composite_npz/'

# Convert date strings to datetime objects
start_date = datetime.strptime(start_date_str, '%Y-%m-%d_%H%M')
end_date = datetime.strptime(end_date_str, '%Y-%m-%d_%H%M')

# Create output directory if needed
os.makedirs(output_dir, exist_ok=True)




# Get list of .pkl files and filter by date range
pkl_files = [
    f for f in os.listdir(input_dir) if f.endswith('.pkl') and 
    start_date <= datetime.strptime(f.split('_')[1] + '_' + f.split('_')[2].split('.')[0], '%Y-%m-%d_%H%M') <= end_date
]

for pkl_file in pkl_files:
    try:
        # Extract datetime from filename
        dt_str = pkl_file.split('_')[1] + '_' + pkl_file.split('_')[2].split('.')[0]
        dt_obj = datetime.strptime(dt_str, '%Y-%m-%d_%H%M')
        npz_time = dt_obj.strftime('%Y%m%d%H%M00')
        
        # Load input data
        filtered_df = pd.read_pickle(os.path.join(input_dir, pkl_file))
        
        # Load NPZ file
        npz_path = f"{npz_base}{npz_time}_conv_wind_composite_data.npz"
        with np.load(npz_path) as data:
            ZH = data['ZH_max']
            rad_shear = data['RAD_SHEAR_LLSD_max']
            KDP = data['KDP_max']
            
        # Process data
        new_df = calculate_metrics(
            filtered_df=filtered_df,
            clons=clons,
            clats=clats,
            ZH=ZH,
            rad_shear=rad_shear,
            KDP=KDP
        )

        # Save results
        output_path = os.path.join(output_dir, pkl_file)
        new_df.to_pickle(output_path)
        print(f"Processed {pkl_file} successfully")

    except Exception as e:
        print(f"Error processing {pkl_file}: {str(e)}")

