import cdsapi
import cfgrib
import pandas as pd
import os
import numpy as np
import xarray as xr
from functools import reduce
import datetime as dt
import multiprocessing
from datetime import datetime, timedelta
from sqlalchemy import create_engine, inspect
import csv  
from scipy.stats import zscore
import matplotlib.pyplot as plt
import seaborn as sns


def weighted_mean(x, weights):
    mask = ~x.isna()
    x = x[mask]
    weights = weights[mask]
    return np.sum(x * weights) / np.sum(weights)

def process_single_day(date_tuple):
    year, month, day = date_tuple
    year_str, month_str, day_str = f"{year}", f"{month:02d}", f"{day:02d}"
    
    # Initialize CDS API client
    client = cdsapi.Client()

    # Define the GRIB file name
    grib_file = f"{year_str}_{month_str}_{day_str}_era5.grib"
    dataset = "reanalysis-era5-single-levels"
    request = {
        "product_type": ["reanalysis"],
        "variable": [
            "2m_dewpoint_temperature",
            "2m_temperature",
            "sea_surface_temperature",
            "total_column_water_vapour",
            "skin_temperature",
            "mean_total_precipitation_rate",
            "total_cloud_cover",
            "runoff",
            "surface_runoff"
        ],
        "year": [str(year)],
        "month": [str(month).zfill(2)],
        "day": [str(day).zfill(2)],
        "time": [
            "00:00", "01:00", "02:00",
            "03:00", "04:00", "05:00",
            "06:00", "07:00", "08:00",
            "09:00", "10:00", "11:00",
            "12:00", "13:00", "14:00",
            "15:00", "16:00", "17:00",
            "18:00", "19:00", "20:00",
            "21:00", "22:00", "23:00"
        ],
        "data_format": "grib",
        "download_format": "unarchived",
        "area": [90, -180, -90, 180],
        "grid": [0.5, 0.5]
    }
    # Download GRIB file
    print(f"Downloading {grib_file}...")
    client.retrieve(dataset, request).download(grib_file)
    # Open GRIB file
    dataset = cfgrib.open_datasets(grib_file)
    # Initialize a list to store DataFrames
    

    dfs = []

    for ds in dataset:
        # Try to drop 'step' and other multi-dim coords if needed
        if "step" in ds.dims:
            ds = ds.mean(dim="step")  # Or choose step=0 or another value

        if "valid_time" in ds.coords:
            ds = ds.drop_vars("valid_time")

        if "number" in ds.coords:
            ds = ds.drop_vars("number")

        try:
            df = ds.to_dataframe().reset_index()
            dfs.append(df)
        except Exception as e:
            print(f"Failed converting a dataset: {e}")

    # Merge all DataFrames on ['time', 'latitude', 'longitude']
    if dfs:
        merged_df = reduce(lambda left, right: pd.merge(
            left, right,
            on=["time", "latitude", "longitude"],
            how="outer",
            suffixes=('', '_dup')  # Prevent merge error on duplicate names
        ), dfs)

        # Drop duplicated columns if created
        merged_df = merged_df.loc[:, ~merged_df.columns.str.endswith('_dup')]

    else:
        merged_df = pd.DataFrame()
        print("No dataframes were created from the datasets.")
    # preprocess the merged DataFrame
    dfm = merged_df.copy()
    dfm['coordinates'] = dfm['latitude'].astype(str) + ',' + dfm['longitude'].astype(str)
    dfm['time'] = pd.to_datetime(dfm['time'], format='%Y-%m-%dT%H:%M:%S.%fZ')
    dfm['day'] = dfm['time'].dt.day
    dfm['month'] = dfm['time'].dt.month
    dfm['year'] = dfm['time'].dt.year
    dfm['hour'] = dfm['time'].dt.hour
    dfm['day of the year'] = dfm['time'].dt.dayofyear
    new_order = ['time', 'day of the year', 'day', 'month', 'year', 'hour', 'latitude', 'longitude', 'coordinates'] + [col for col in dfm.columns if col not in ['time', 'day', 'month', 'year', 'hour', 'latitude', 'longitude', 'coordinates']]
    dfm = dfm[new_order]
    #drop date if it comes from different day
    dfm.sort_values(by='time')
    if dfm['day'].iloc[0] != dfm['day'].iloc[-1]:
        dfm.dropna(subset=['t2m'], inplace=True)
    
    # Define log path
    log_path = r"C:\Users\dmoli\Documents\Coding\Weathercast_project\date_log_global.csv"
    dfm['time'] = pd.to_datetime(dfm['time'])
    dfm = dfm.sort_values(['coordinates', 'time'])

    # Start tracking broken columns
    broken_columns = []

    # Start log list
    log_entries = []

    # Check missing data
    missing_data = dfm.isnull()

    for column in dfm.columns:
        missing_count = missing_data[column].sum()
        if isinstance(missing_count, pd.Series):
            missing_count = missing_count.sum()
        missing_count = int(missing_count)

        if missing_count > 0:
            timestamp = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # Columns that can tolerate up to 3 full missing hours
            relaxed_cols = [
                'tcwv','t2m', 'd2m', 'skt', 'tcc' 
                                        ]

            if column not in relaxed_cols:
                if missing_count > 1433520:
                    log_entries.append([timestamp, column, missing_count, 'too many missing values'])
                    broken_columns.append(column)
            else:
                ratio = round(missing_count / 65160, 2)
                if ratio > 3:
                    log_entries.append([timestamp, column, missing_count, f'missing > 3 days ({ratio})'])
                    broken_columns.append(column)

    # Save log if needed
    if log_entries:
        log_df = pd.DataFrame(log_entries, columns=["timestamp", "variable", "missing_count", "note"])
        if os.path.exists(log_path):
            log_df.to_csv(log_path, mode='a', header=False, index=False)
        else:
            log_df.to_csv(log_path, index=False)

    
    # process information to be stored in the database
    

    dfm['t2m'] = (dfm['t2m'] - 273.15).round(2)
    dfm['d2m'] = (dfm['d2m'] - 273.15).round(2)
    dfm['skt'] = (dfm['skt'] - 273.15).round(2)
    dfm['lat_weight'] = np.cos(np.deg2rad(dfm['latitude']))
    
    weather = [ 't2m', 'd2m', 'sst', 'skt', 'tcc', 'sro', 'ro', 'avg_tprate', 'tcwv']


    # global statistics for each weather variable
    common_stats = [weighted_mean, 'min', 'max', 'std', 'median', lambda x: x.max() - x.min()]
    #weighted mean since we have latitudes and longitudes and pole data is not evenly distributed
    summary_data = {}
    for weather_var in weather:
        for stat in common_stats:
            key = f"{weather_var}_{stat.__name__}" if callable(stat) else f"{weather_var}_{stat}"
            
            if stat == weighted_mean:
                # Call weighted mean with weights
                summary_data[key] = weighted_mean(dfm[weather_var], dfm['lat_weight'])
            else:
                # Regular aggregation
                summary_data[key] = dfm[weather_var].agg(stat)
           
                

    summary_df = pd.DataFrame([summary_data])
    summary_df['Date'] = dfm['time'].iloc[0]
    summary_df['Date'] = pd.to_datetime(summary_df['Date'])
    summary_df['day_of_year'] = summary_df['Date'].dt.dayofyear
    summary_df['coord_min_temp'] = dfm.loc[dfm['t2m'].idxmin(), 'coordinates']
    summary_df['coord_max_temp'] = dfm.loc[dfm['t2m'].idxmax(), 'coordinates']
    summary_df['coord_min_time'] = dfm.loc[dfm['t2m'].idxmin(), 'hour']
    summary_df['coord_max_time'] = dfm.loc[dfm['t2m'].idxmax(), 'hour']
    summary_df['ocean_min_temp'] = dfm.loc[dfm['sst'].idxmin(), 'coordinates']
    summary_df['ocean_max_temp'] = dfm.loc[dfm['sst'].idxmax(), 'coordinates']
    summary_df['ocean_min_time'] = dfm.loc[dfm['sst'].idxmin(), 'hour']
    summary_df['ocean_max_time'] = dfm.loc[dfm['sst'].idxmax(), 'hour']
    new_order = ['Date', 'day_of_year'] + [col for col in summary_df.columns if col not in ['Date', 'day_of_year']]
    summary_df = summary_df[new_order]

    
    # Save to DB
    # MySQL Connection
    user = 'root'
    password = 'Hamilton1186!'
    host = '127.0.0.1'
    port = '3306'
    db = 'weatherdb'
    table_name = 'weather_summary_global'

    # Create SQLAlchemy engine
    engine = create_engine(f"mysql+pymysql://{user}:{password}@{host}:{port}/{db}")

    # Check if table exists
    inspector = inspect(engine)
    if not inspector.has_table(table_name):
        # Create table by writing empty df with same schema
        summary_df.head(0).to_sql(name=table_name, con=engine, if_exists='replace', index=False)
        print(f"Table '{table_name}' created.")
    else:
        print(f"Table '{table_name}' already exists. Skipping creation.")

    # Append data
    summary_df.to_sql(name=table_name, con=engine, if_exists='append', index=False)
    print(f"Data inserted into '{table_name}'.")

    # Dispose of the engine to close the connection
    engine.dispose()
    # Delete GRIB file if it exists
    if os.path.exists(grib_file):
        try:
            os.remove(grib_file)
            print(f"Deleted {grib_file}")
        except Exception as e:
            print(f"Error deleting {grib_file}: {e}")

    print("DataFrame successfully stored in the MySQL database!")
    # Define the log path and name
    log_path = r"C:\Users\dmoli\Documents\Coding\Weathercast_project\storage log.csv"

    # Get the current timestamp and data date
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    data_date = summary_df['Date'].iloc[0]  # Assuming this is the date of the data

    # Build the log row: [data_date, timestamp, count_col1, count_col2, ..., count_colN]
    log_row = [data_date, timestamp] 

    # Write to CSV (append mode, no headers)
    with open(log_path, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(log_row)

    # Delete GRIB file if it exists
    if os.path.exists(grib_file):
        try:
            os.remove(grib_file)
            print(f"Deleted {grib_file}")
        except Exception as e:
            print(f"Error deleting {grib_file}: {e}")            
    print(f"Processing {year}-{month:02d}-{day:02d}")
    pass

def get_all_dates(start_date="1940-01-03", end_date="2025-05-23"):
    if end_date is None:
        end_date = datetime.today().strftime('%Y-%m-%d')
    start = datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.strptime(end_date, '%Y-%m-%d')
    delta = end - start
    return [(d.year, d.month, d.day) for d in (start + timedelta(days=i) for i in range(delta.days + 1))]

if __name__ == "__main__":
    all_dates = get_all_dates()
    pool_size = 6  # Number of parallel processes

    with multiprocessing.Pool(pool_size) as pool:
        pool.map(process_single_day, all_dates)

 