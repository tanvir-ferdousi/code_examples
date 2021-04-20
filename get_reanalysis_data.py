# Import Required libraries
import netCDF4 as nc
import os
import urllib.request as req
import pandas as pd
from datetime import datetime

# Function definitions

def download_netcdf_file(var_name, year, var_map, directory, ftp_loc):
    # Download raw data file (.nc)
    file_name = var_map[var_name] + '.' + str(year) + '.nc'
    file_path = os.path.join(directory, file_name)
    
    url = ftp_loc+ file_name

    if not os.path.exists(directory):
        print("Directory not found!")
        os.mkdir(directory)
        print("Created new folder: data")

    file_path = os.path.join(directory, file_name)

    if not os.path.exists(file_path):
        req.urlretrieve(url, file_path)
        print('File', file_name, 'downloaded.')
    else:
        print('File', file_name, 'already exists.')

def read_netcdf_file(lat, lon, date, var_name, var_map, directory):
    
    date_stamp = pd.to_datetime(date)
    day = date_stamp.dayofyear
    year = date_stamp.year
    
    file_name = var_map[var_name] + '.' + str(year) + '.nc'
    file_path = os.path.join(directory, file_name)
    
    if not os.path.exists(file_path):
        print('File', file_name, 'not found.')
        return
    
    dataset = nc.Dataset(file_path)
    
    if lon < 0:
        lon = lon + 360
    
    data_values = dataset[var_name][:]
    
    lat_values = dataset['lat'][:]
    lon_values = dataset['lon'][:]

    time_idx = day - 1
    lat_idx =find_index(lat_values, lat)
    lon_idx =find_index(lon_values, lon)
    
    return data_values[time_idx, lat_idx, lon_idx]

def find_index(var_array, var_value):
    return min(range(len(var_array)), key=lambda i: abs(var_array[i]-var_value))
 


## Example Usage

# Read data

# var descriptions are given below (air, rhum, pres, pr_wtr)
# Surface Air Temperature: air
# Relative Humidity: rhum
# Surface Level Pressure: pres
# Precipitable Water: pr_wtr

# Source: https://psl.noaa.gov/data/gridded/data.ncep.reanalysis.surface.html
var_map = {
'air': 'air.sig995',
'rhum': 'rhum.sig995',
'pres': 'pres.sfc',
'pr_wtr': 'pr_wtr.eatm'
}

ftp_loc = 'ftp://ftp2.psl.noaa.gov/Datasets/ncep.reanalysis.dailyavgs/surface/'


var_name = 'air'
lat = -22.876652118200003
lon = -43.227875125
date = '2019-08-25'

year = date.split('-')[0]

# Uses a 'data' folder in your current directory to store .nc files
directory = os.path.join(os.getcwd(), 'data')

# download .nc file containing the specified var for the specified year
download_netcdf_file(var_name, year, var_map, directory, ftp_loc)

# read value
var_value = read_netcdf_file(lat, lon, date, var_name, var_map, directory)

# print value
print('The value for the variable', '\''+var_name+'\'', 'is', var_value)
