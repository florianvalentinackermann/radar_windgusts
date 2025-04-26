# Import packages

import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import pyart
import glob
import matplotlib.patheffects as path_effects
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap
from pyproj import Transformer
import radlib
import os
import h5py
import matplotlib.dates as mdates
from scipy.ndimage import gaussian_filter1d
from cartopy.feature import NaturalEarthFeature
import xmltodict, geopandas, geojson, xml #xml and json do not exist
from datetime import datetime, timedelta, timezone
import geopy.distance
from datetime import datetime, timedelta
import numpy.matlib as npm
import copy
from scipy.signal import convolve2d
from astropy.convolution import convolve
import scipy.ndimage as ndi
import re
from skimage.draw import polygon

from pprint import pprint
from pysteps import io, motion, rcparams
from pysteps.utils import conversion, transformation
from pysteps.visualization import plot_precip_field, quiver

import json

from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.font_manager as fm

import polars as pl

import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling

import geopandas as gpd
from shapely.geometry import Point

from mpl_toolkits.mplot3d import Axes3D

from shapely.geometry import Polygon


from geopandas import GeoDataFrame
import datetime

import matplotlib.patches as patches

import geojson as gs


os.environ["library_metranet_path"] = "/store_new/mch/msrad/idl/lib/radlib4/" # needed for pyradlib
os.environ["METRANETLIB_PATH"] = "/store_new/mch/msrad/idl/lib/radlib4/" # needed for pyart_mch


# Data Imports, processing, norming, etc.


# GPSConverter for coordinate transformation
class GPSConverter(object):
    '''
    GPS Converter class which is able to perform convertions between the 
    CH1903 and WGS84 system.
    '''
    # Convert CH y/x/h to WGS height
    def CHtoWGSheight(self, y, x, h):
        # Axiliary values (% Bern)
        y_aux = (y - 600000) / 1000000
        x_aux = (x - 200000) / 1000000
        h = (h + 49.55) - (12.60 * y_aux) - (22.64 * x_aux)
        return h

    # Convert CH y/x to WGS lat
    def CHtoWGSlat(self, y, x):
        # Axiliary values (% Bern)
        y_aux = (y - 600000) / 1000000
        x_aux = (x - 200000) / 1000000
        lat = (16.9023892 + (3.238272 * x_aux)) + \
                - (0.270978 * pow(y_aux, 2)) + \
                - (0.002528 * pow(x_aux, 2)) + \
                - (0.0447 * pow(y_aux, 2) * x_aux) + \
                - (0.0140 * pow(x_aux, 3))
        # Unit 10000" to 1" and convert seconds to degrees (dec)
        lat = (lat * 100) / 36
        return lat

    # Convert CH y/x to WGS long
    def CHtoWGSlng(self, y, x):
        # Axiliary values (% Bern)
        y_aux = (y - 600000) / 1000000
        x_aux = (x - 200000) / 1000000
        lng = (2.6779094 + (4.728982 * y_aux) + \
                + (0.791484 * y_aux * x_aux) + \
                + (0.1306 * y_aux * pow(x_aux, 2))) + \
                - (0.0436 * pow(y_aux, 3))
        # Unit 10000" to 1" and convert seconds to degrees (dec)
        lng = (lng * 100) / 36
        return lng

    # Convert decimal angle (° dec) to sexagesimal angle (dd.mmss,ss)
    def DecToSexAngle(self, dec):
        degree = int(math.floor(dec))
        minute = int(math.floor((dec - degree) * 60))
        second = (((dec - degree) * 60) - minute) * 60
        return degree + (float(minute) / 100) + (second / 10000)
		
    # Convert sexagesimal angle (dd.mmss,ss) to seconds
    def SexAngleToSeconds(self, dms):
        degree = 0 
        minute = 0 
        second = 0
        degree = math.floor(dms)
        minute = math.floor((dms - degree) * 100)
        second = (((dms - degree) * 100) - minute) * 100
        return second + (minute * 60) + (degree * 3600)

    # Convert sexagesimal angle (dd.mmss) to decimal angle (degrees)
    def SexToDecAngle(self, dms):
        degree = 0
        minute = 0
        second = 0
        degree = math.floor(dms)
        minute = math.floor((dms - degree) * 100)
        second = (((dms - degree) * 100) - minute) * 100
        return degree + (minute / 60) + (second / 3600)
    
    # Convert WGS lat/long (° dec) and height to CH h
    def WGStoCHh(self, lat, lng, h):
        lat = self.DecToSexAngle(lat)
        lng = self.DecToSexAngle(lng)
        lat = self.SexAngleToSeconds(lat)
        lng = self.SexAngleToSeconds(lng)
        # Axiliary values (% Bern)
        lat_aux = (lat - 169028.66) / 10000
        lng_aux = (lng - 26782.5) / 10000
        h = (h - 49.55) + (2.73 * lng_aux) + (6.94 * lat_aux)
        return h

    # Convert WGS lat/long (° dec) to CH x
    def WGStoCHx(self, lat, lng):
        lat = self.DecToSexAngle(lat)
        lng = self.DecToSexAngle(lng)
        lat = self.SexAngleToSeconds(lat)
        lng = self.SexAngleToSeconds(lng)
        # Axiliary values (% Bern)
        lat_aux = (lat - 169028.66) / 10000
        lng_aux = (lng - 26782.5) / 10000
        x = ((200147.07 + (308807.95 * lat_aux) + \
            + (3745.25 * pow(lng_aux, 2)) + \
            + (76.63 * pow(lat_aux,2))) + \
            - (194.56 * pow(lng_aux, 2) * lat_aux)) + \
            + (119.79 * pow(lat_aux, 3))
        return x

	# Convert WGS lat/long (° dec) to CH y
    def WGStoCHy(self, lat, lng):
        lat = self.DecToSexAngle(lat)
        lng = self.DecToSexAngle(lng)
        lat = self.SexAngleToSeconds(lat)
        lng = self.SexAngleToSeconds(lng)
        # Axiliary values (% Bern)
        lat_aux = (lat - 169028.66) / 10000
        lng_aux = (lng - 26782.5) / 10000
        y = (600072.37 + (211455.93 * lng_aux)) + \
            - (10938.51 * lng_aux * lat_aux) + \
            - (0.36 * lng_aux * pow(lat_aux, 2)) + \
            - (44.54 * pow(lng_aux, 3))
        return y

    def LV03toWGS84(self, east, north, height):
        '''
        Convert LV03 to WGS84 Return a array of double that contain lat, long,
        and height
        '''
        d = []
        d.append(self.CHtoWGSlat(east, north))
        d.append(self.CHtoWGSlng(east, north))
        d.append(self.CHtoWGSheight(east, north, height))
        return d
        
    def WGS84toLV03(self, latitude, longitude, ellHeight):
        '''
        Convert WGS84 to LV03 Return an array of double that contaign east,
        north, and height
        '''
        d = []
        d.append(self.WGStoCHy(latitude, longitude))
        d.append(self.WGStoCHx(latitude, longitude))
        d.append(self.WGStoCHh(latitude, longitude, ellHeight))
        return d

# c_transform
def c_transform(lon,lat):
    """
    transforms arrays of lat/lon to chx/chy

    Parameters
    ----------
    lon : float
        longitude.
    lat : float
        latitude.

    Returns
    -------
    chx : float
        chx in m.
    chy : float
        chy in m.

    """
    converter = GPSConverter()
    chx=np.zeros([len(lon)])
    chy=np.zeros([len(lon)])
    for n in range(len(lon)):
        chx[n],chy[n],z=converter.WGS84toLV03(lat[n], lon[n], 0)
    return chx,chy

# transform_c 
def transform_c(chx,chy):
    """
    transforms arrays of chx/chy to lat/lon
    
    Parameters
    -------
    chx : float
        chx in m.
    chy : float
        chy in m.
        
    Returns
    ----------
    lon : float
        longitude.
    lat : float
        latitude.



    """
    converter = GPSConverter()
    lon=np.zeros([len(chx)])
    lat=np.zeros([len(chy)])
    for n in range(len(lon)):
        lat[n], lon[n],z=converter.LV03toWGS84(chx[n],chy[n], 0)
    return lon,lat


from datetime import datetime, timedelta, timezone


# Define TRT reading function
def read_TRT(path, file=0, ttime=0):
    """
    Read .trt or .json file containing TRT output
    Returns dataframe with attributes and gridded TRT cells

    Parameters
    ----------

    path : string
        path, where to look for files.
    file: string
        filename
    ttime : string
        timestep to find files for.
    Requires either filename or timestep
   
    Returns
    -------
    trt_df : dataframe
        TRT cells and attributes of the timestep.
    cells: list
        Gridded TRT cells per timestep
    timelist: list
        timesteps

    """
   
    o_x=255000
    o_y=-160000
    lx=710; ly=640
    cells=np.zeros([ly,lx])
    if file == 0:
        file=glob.glob(path["lomdata"]+'*'+ttime+'*json*')
        if len(file)>0: flag=1
        else:
            file=glob.glob(path["lomdata"]+'*'+ttime+'*'+'.trt')[0]
            flag=0
    else:
        if 'json' in file: flag=1; ttime=file[-20:-11]
        else: flag=0; ttime=file[-15:-6]
        file=[file]
   
    if flag==1:
        with open(file[0]) as f: gj = geojson.FeatureCollection(gs.load(f))
        trt_df=geopandas.GeoDataFrame.from_features(gj['features'])
        if len(trt_df)>0:
          # print(trt_df.lon.values.astype(float))
          chx, chy = c_transform(trt_df.lon.values.astype(float),trt_df.lat.values.astype(float))
          trt_df['chx']=chx.astype(str); trt_df['chy']=chy.astype(str)
          for n in range(len(trt_df)):
              lon,lat=trt_df.iloc[n].geometry.boundary.xy
              # print(trt_df.iloc[n])
              chx, chy = c_transform(lon,lat)
              # trt_df.iloc[n]['chx']=chx.astype(str); trt_df.iloc[n]['chy']=chy.astype(str)
              #transform.c_transform(trt_df.iloc[n].lon.values,trt_df.iloc[n].lat.values)
              ix=np.round((chx-o_x)/1000).astype(int)
              iy=np.round((chy-o_y)/1000).astype(int)
              rr, cc = polygon(iy, ix, cells.shape)
              # print(lat,lon,chx,chy,ix,iy)
              cells[rr,cc]=int(trt_df.traj_ID.iloc[n]);
        else: cells=[]
    else:
        data = pd.read_csv(file).iloc[8:]
        headers = pd.read_csv(file).iloc[7:8].iloc[0][0].split()
        trt_df = pd.DataFrame()
        geometries = []  # New list to store geometries

        for n in range(len(data)):
            # ... (existing code remains the same)
            t = data.iloc[n].str.split(';', expand=True)
            trt_df.loc[n, 'traj_ID'] = int(t[0].values)
            trt_df.loc[n, 'yyyymmddHHMM'] = str(t[1].values[0])  # Assign as string
            trt_df.loc[n, 'lon'] = t[2].values.astype(float)
            trt_df.loc[n, 'lat'] = t[3].values.astype(float)
            trt_df.loc[n, 'ell_L'] = t[4].values.astype(float)
            trt_df.loc[n, 'ell_S'] = t[5].values.astype(float)
            trt_df.loc[n, 'ell_or'] = t[6].values.astype(float)
            trt_df.loc[n, 'area'] = t[7].values.astype(float)
            trt_df.loc[n, 'vel_x'] = t[8].values.astype(float)
            trt_df.loc[n, 'vel_y'] = t[9].values.astype(float)
            trt_df.loc[n, 'det'] = t[10].values.astype(float)
            trt_df.loc[n, 'RANKr'] = t[11].values.astype(float)
            trt_df.loc[n, 'CG-'] = t[12].values.astype(float)
            trt_df.loc[n, 'CG+'] = t[13].values.astype(float)
            trt_df.loc[n, 'CG'] = t[14].values.astype(float)
            trt_df.loc[n, '%CG+'] = t[15].values.astype(float)
            trt_df.loc[n, 'ET45'] = t[16].values.astype(float)
            trt_df.loc[n, 'ET45m'] = t[17].values.astype(float)
            trt_df.loc[n, 'ET15'] = t[18].values.astype(float)
            trt_df.loc[n, 'ET15m'] = t[19].values.astype(float)
            trt_df.loc[n, 'VIL'] = t[20].values.astype(float)
            trt_df.loc[n, 'maxH'] = t[21].values.astype(float)
            trt_df.loc[n, 'maxHm'] = t[22].values.astype(float)
            trt_df.loc[n, 'POH'] = t[23].values.astype(float)
            trt_df.loc[n, 'MESHS'] = t[24].values.astype(float)
            trt_df.loc[n, 'Dvel_x'] = t[25].values.astype(float)
            trt_df.loc[n, 'Dvel_y'] = t[26].values.astype(float)
            chx, chy = c_transform([trt_df.loc[n, 'lon']], [trt_df.loc[n, 'lat']])
            ix = np.round((chx - o_x) / 1000).astype(int)
            if ix >= 710: ix = 709
            iy = np.round((chy - o_y) / 1000).astype(int)
            if iy >= 640: iy = 639
            n2 = 27
            #if int(ttime) >= 221520631: n2 = 82
            tt = np.array(t)[0, n2:-1]
            tt = np.reshape(tt, [int(len(tt) / 2), 2])
            trt_df.loc[n, 'chx'] = chx
            trt_df.loc[n, 'chy'] = chy
            lat = tt[:, 1].astype(float)
            lon = tt[:, 0].astype(float)
            chx, chy = c_transform(lon, lat)
            ix = np.round((chx - o_x) / 1000).astype(int)
            iy = np.round((chy - o_y) / 1000).astype(int)
            rr, cc = polygon(iy, ix, cells.shape)
            cells[rr, cc] = int(t[0].values)
            # Create polygon for this cell
            polygon_coords = list(zip(lon, lat))
            cell_polygon = Polygon(polygon_coords)
            geometries.append(cell_polygon)
        
        # Add geometry column to trt_df
        trt_df['geometry'] = geometries
    
        # Convert trt_df to GeoDataFrame
        trt_df = gpd.GeoDataFrame(trt_df, geometry='geometry', crs="EPSG:4326")

    timelist=[str(ttime)]
    return trt_df, [cells], timelist



import zipfile


def process_gust_markers(valid_date3, valid_time3, extraction_dir):
    # Convert date/time parameters
    date_obj = datetime.strptime(valid_date3, '%Y-%m-%d')
    year_last_two = date_obj.strftime('%y')
    day_of_year = date_obj.timetuple().tm_yday
    valid_date4 = f"{year_last_two}{day_of_year:03d}"
    valid_time4 = valid_time3 + '0'

    # Find matching TRT data file in extraction directory
    target_pattern = f"CZC{valid_date4}{valid_time4}"
    trt_data_file = next((f for f in os.listdir(extraction_dir) 
                         if f.startswith(target_pattern)), None)

    if not trt_data_file:
        return pd.DataFrame(columns=['geometry', 'Age', 'CS Marker', 'STA Marker', 
                                   'ESWD Marker', 'STA Speed', 'Gust_Flag'])

    trt_data_path = extraction_dir
    
    # Attempt to read TRT data and handle errors gracefully
    try:
        trt_df, cells_list, timelist = read_TRT({"lomdata": trt_data_path}, 
                                                ttime=valid_date4+valid_time4)
    except Exception as e:
        # Log the error if needed (optional)
        print(f"Error reading TRT data: {e}")
        # Return an empty DataFrame with required columns
        return pd.DataFrame(columns=['geometry', 'Age', 'CS Marker', 'STA Marker', 
                                     'ESWD Marker', 'STA Speed', 'Gust_Flag'])
    
    # Load Gust Markers dataframe
    df = pd.read_pickle("/scratch/mch/fackerma/orders/Gust_Markers/Gust_Markers_3.pkl")
        
    if trt_df.empty:
        # Return an empty DataFrame with required columns if TRT data is empty
        empty_df = pd.DataFrame(columns=['geometry', 'Age', 'CS Marker', 'STA Marker', 'ESWD Marker', 'STA Speed', 'Gust_Flag'])
        trt_df = empty_df
        return trt_df
        
    # Alternative: Extract the larges 5 cities based on popdense > 2000 / km sqared
    file_path_swiss_cities = "polygons_wgs84_1000.gpkg"
    # Read the GeoPackage into a GeoDataFrame
    swiss_cities_gdf = gpd.read_file(file_path_swiss_cities)

    # Ensure yyyymmddHHMM and traj_ID are strings (to avoid float conversion issues)
    trt_df['yyyymmddHHMM'] = trt_df['yyyymmddHHMM'].astype(str)
    trt_df['yyyymmddHHMM'] = trt_df['yyyymmddHHMM'].str.strip()  # Remove leading/trailing spaces
        
    trt_df['traj_ID'] = trt_df['traj_ID'].astype(str)

    # Convert yyyymmddHHMM to datetime
    trt_df['current_time'] = pd.to_datetime(trt_df['yyyymmddHHMM'], format='%Y%m%d%H%M')

    # Extract birth time from the first 12 digits of traj_ID and convert to datetime
    trt_df['birth_time'] = pd.to_datetime(trt_df['traj_ID'].str[:12], format='%Y%m%d%H%M')

    # Calculate age in minutes
    trt_df['Age'] = (trt_df['current_time'] - trt_df['birth_time']).dt.total_seconds() / 60

    # Optionally, drop helper columns
    trt_df.drop(columns=['current_time', 'birth_time'], inplace=True)
        
    # Convert TRT DataFrame to GeoDataFrame and process buffers
    trt_gdf = GeoDataFrame(trt_df, geometry='geometry', crs="EPSG:4326")

    trt_gdf_projected = trt_gdf.to_crs(epsg=32632)
    trt_gdf_projected['buffer_geometry'] = trt_gdf_projected.geometry.buffer(10000)
    trt_gdf_with_buffer = trt_gdf_projected.to_crs(epsg=4326)
        
    trt_gdf_small_buffer = trt_gdf.to_crs(epsg=32632)
    trt_gdf_small_buffer['buffer_geometry'] = trt_gdf_small_buffer.geometry.buffer(5000)
    trt_gdf_small_buffer = trt_gdf_small_buffer.to_crs(epsg=4326)
    trt_gdf_small_buffer['buffer_geometry'] = trt_gdf_small_buffer['buffer_geometry'].to_crs(epsg=4326)

    # Set up datetime for filtering
    datetime_string = f"{valid_date3} {valid_time3[:2]}:{valid_time3[2:]}:00"
    target_datetime = pd.to_datetime(datetime_string)
    time_window_cs = pd.Timedelta(minutes=2.5)
    time_window_sta = pd.Timedelta(minutes=5)
    time_window_eswd = pd.Timedelta(minutes=2.5)

    # Filter and process CS and STA data
    df_cs = df[df['Source'] == 'CS']
    df_sta = df[df['Source'] == 'STA'].copy()
    df_sta['Time'] = df_sta['Time'] - pd.Timedelta(minutes=5)
    df_eswd = df[df['Source'] == 'ESWD']

    filtered_df_cs = df_cs[(df_cs['Time'] >= target_datetime - time_window_cs) & 
                           (df_cs['Time'] <= target_datetime + time_window_cs)]
    filtered_df_sta = df_sta[(df_sta['Time'] >= target_datetime - time_window_sta) & 
                            (df_sta['Time'] <= target_datetime + time_window_sta)]
    filtered_df_eswd = df_eswd[(df_eswd['Time'] >= target_datetime - time_window_eswd) & 
                            (df_eswd['Time'] <= target_datetime + time_window_eswd)]

    # Convert to GeoDataFrames
    gust_gdf_cs = GeoDataFrame(filtered_df_cs, 
                               geometry=gpd.points_from_xy(filtered_df_cs.Longitude, filtered_df_cs.Latitude),
                               crs="EPSG:4326")
    gust_gdf_sta = GeoDataFrame(filtered_df_sta, 
                               geometry=gpd.points_from_xy(filtered_df_sta.Longitude, filtered_df_sta.Latitude),
                               crs="EPSG:4326")
    gust_gdf_eswd = GeoDataFrame(filtered_df_eswd, 
                                geometry=gpd.points_from_xy(filtered_df_eswd.Longitude, filtered_df_eswd.Latitude),
                                crs="EPSG:4326")

    # After creating trt_gdf_with_buffer:
    trt_gdf_buffer = trt_gdf_with_buffer.to_crs(epsg=4326)  # Ensure CRS matches gust_gdf_sta
    trt_gdf_buffer.set_geometry('buffer_geometry', inplace=True)
    trt_gdf_buffer['buffer_geometry'] = trt_gdf_buffer['buffer_geometry'].to_crs(epsg=4326)

    # Perform spatial joins for CS gusts
    #trt_gdf_buffer = trt_gdf_with_buffer.copy()
    #trt_gdf_buffer.set_geometry('buffer_geometry', inplace=True)
    trt_gdf_small_buffer.set_geometry('buffer_geometry', inplace=True)

    # For CS data
    matched_gusts_cs = gpd.sjoin(gust_gdf_cs, trt_gdf_small_buffer, how="inner", predicate="within", rsuffix='_trt')
    matched_gusts_cs = matched_gusts_cs.sort_values(by=['Age', 'RANKr'], ascending=[False, False]).drop_duplicates(subset=['Longitude', 'Latitude', 'Time'], keep='first')

    # For STA data
    matched_gusts_sta = gpd.sjoin(gust_gdf_sta, trt_gdf_buffer, how="inner", predicate="within", rsuffix='_trt')
    matched_gusts_sta = matched_gusts_sta.sort_values(by=['Age', 'RANKr'], ascending=[False, False]).drop_duplicates(subset=['Longitude', 'Latitude', 'Time'], keep='first')

    # For ESWD data
    matched_gusts_eswd = gpd.sjoin(gust_gdf_eswd, trt_gdf_small_buffer, how="inner", predicate="within", rsuffix='_trt')
    matched_gusts_eswd = matched_gusts_eswd.sort_values(by=['Age', 'RANKr'], ascending=[False, False]).drop_duplicates(subset=['Longitude', 'Latitude', 'Time'], keep='first')


    # CS markers
    cs_counts = matched_gusts_cs['traj_ID'].value_counts().to_dict()
    trt_df['CS Marker'] = trt_df['traj_ID'].map(cs_counts).fillna(0).astype(int)

    # STA markers
    sta_counts = matched_gusts_sta['traj_ID'].value_counts().to_dict()
    trt_df['STA Marker'] = trt_df['traj_ID'].map(sta_counts).fillna(0).astype(int)

    # ESWD markers
    eswd_counts = matched_gusts_eswd['traj_ID'].value_counts().to_dict()
    trt_df['ESWD Marker'] = trt_df['traj_ID'].map(eswd_counts).fillna(0).astype(int)

    # Initialize 'STA Speed' column
    trt_df['STA Speed'] = None

    # Collect wind speeds for each traj_ID
    sta_speeds = matched_gusts_sta.groupby('traj_ID')['Attribute'].apply(list).to_dict()

    # Map the collected wind speeds to the 'STA Speed' column
    trt_df['STA Speed'] = trt_df['traj_ID'].map(sta_speeds)

    # Convert lists with a single element to just that element
    trt_df['STA Speed'] = trt_df['STA Speed'].apply(lambda x: x[0] if isinstance(x, list) and len(x) == 1 else x)


    # Create a new geometry column in trt_gdf based on lon and lat
    trt_gdf['point_geometry'] = trt_gdf.apply(lambda row: Point(row['lon'], row['lat']), axis=1)

    # Perform the intersection check using the new point_geometry column
    touches_swiss_cities = trt_gdf['point_geometry'].apply(
        lambda point: swiss_cities_gdf.intersects(point).any()
    )
    
    # Assign 'Gust_Flag' based on conditions
    trt_df['Gust_Flag'] = np.where(
        (trt_df['CS Marker'] > 1) | (trt_df['STA Marker'] > 0) | (trt_df['ESWD Marker'] > 0),
        'Yes',
        np.where(
            (touches_swiss_cities) &
            (target_datetime.time() >= datetime.strptime("04:00:00", "%H:%M:%S").time()) &
            (target_datetime.time() <= datetime.strptime("20:00:00", "%H:%M:%S").time()) &
            (target_datetime.year in [2022, 2023]) &
            (trt_df['CS Marker'] == 0) &
            (trt_df['STA Marker'] == 0) &
            (trt_df['ESWD Marker'] == 0),
            'No',
            '-'
        )
    )
    
    return trt_df




# Define the date range
start_date = datetime.strptime('2019-05-01', '%Y-%m-%d')
end_date = datetime.strptime('2019-10-31', '%Y-%m-%d')  
time_delta = timedelta(minutes=5)

# Define paths
extraction_dir = "/scratch/mch/fackerma/orders/TRT_Unzip/"
output_dir = "/scratch/mch/fackerma/orders/TRT_processing_3/2019"

current_date = start_date
while current_date <= end_date:
    # Prepare date parameters
    valid_date3 = current_date.strftime('%Y-%m-%d')
    date_obj = datetime.strptime(valid_date3, '%Y-%m-%d')
    year_last_two = date_obj.strftime('%y')
    day_of_year = date_obj.timetuple().tm_yday
    valid_date4 = f"{year_last_two}{day_of_year:03d}"
    
    # Define ZIP file path
    base_path = f"/store_new/mch/msrad/radar/swiss/data/{date_obj.year}/{valid_date4}/"
    zip_file_name = f"TRTC{valid_date4}.zip"
    zip_file_path = os.path.join(base_path, zip_file_name)
    
    try:
        # Unzip once per day
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(extraction_dir)
        print(f"Unzipped {zip_file_name} for daily processing")

        # Process all time steps for this day
        current_time = datetime.combine(current_date, datetime.strptime('0000', '%H%M').time())
        end_time = datetime.combine(current_date, datetime.strptime('2355', '%H%M').time())

        while current_time <= end_time:
            valid_time3 = current_time.strftime('%H%M')
            #print(f"Processing for date: {valid_date3}, time: {valid_time3}")
            
            # Modified process_gust_markers now takes extraction_dir as argument
            trt_df = process_gust_markers(valid_date3, valid_time3, extraction_dir)
            
            # Save results
            filename = f'TRT_{valid_date3}_{valid_time3}.pkl'
            trt_df.to_pickle(os.path.join(output_dir, filename))
            
            current_time += time_delta

    except FileNotFoundError:
        print(f"ZIP file not found: {zip_file_path}")
        continue  # Skip to next date if file missing

    finally:
        # Cleanup after daily processing
        for file_name in os.listdir(extraction_dir):
            if file_name.startswith("CZC"):
                file_path = os.path.join(extraction_dir, file_name)
                try:
                    os.remove(file_path)
                    print(f"Deleted temporary file: {file_path}")
                except Exception as e:
                    print(f"Error deleting {file_path}: {e}")

    current_date += timedelta(days=1)

print("Processing complete for all dates and times.")