import cdsapi
import cfgrib
import pandas as pd
import os
import numpy as np
import xarray as xr
from functools import reduce
import datetime as dt
from scipy.stats import skew, kurtosis
import multiprocessing
from datetime import datetime, timedelta
from sqlalchemy import create_engine, inspect
import csv  
from scipy.stats import zscore
import matplotlib.pyplot as plt
import seaborn as sns
from skyfield.api import load
from skyfield.framelib import ecliptic_frame
import region_mask
from skyfield.api import load, Topos
from skyfield import almanac


#function to get season based on region
def get_region_from_mask(lat, lon, mask_da):
    lat_idx = mask_da.lat.sel(lat=lat, method='nearest').values
    lon_idx = mask_da.lon.sel(lon=lon, method='nearest').values
    return mask_da.sel(lat=lat_idx, lon=lon_idx).item()


# function to get the season based on the day of the year and latitude
def get_season(day_of_year, year, latitude=None):
    planets = load('de421.bsp')
    sun = planets['sun']
    earth = planets['earth']
    ts = load.timescale()
    t = ts.utc(year, 1, day_of_year)
    astrometric = earth.at(t).observe(sun)
    lat, lon, distance = astrometric.frame_latlon(ecliptic_frame)
    lon_deg = lon.degrees % 360
    
    # Northern Hemisphere seasons
    if 0 <= lon_deg < 90:
        season = 'Spring'
    elif 90 <= lon_deg < 180:
        season = 'Summer'
    elif 180 <= lon_deg < 270:
        season = 'Autumn'
    else:
        season = 'Winter'
    
    # Adjust for Southern Hemisphere if latitude is provided
    if latitude is not None and latitude < 0:
        season_map = {'Spring': 'Autumn', 'Summer': 'Winter', 'Autumn': 'Spring', 'Winter': 'Summer'}
        return season_map[season]
    return season

def safe_skew(x):
    return skew(x) if x.var() > 1e-10 else np.nan

def safe_kurtosis(x):
    return kurtosis(x, fisher=True) if x.var() > 1e-10 else np.nan


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
            "10m_u_component_of_wind",
            "10m_v_component_of_wind",
            "2m_dewpoint_temperature",
            "2m_temperature",
            "mean_sea_level_pressure",
            "sea_surface_temperature",
            "surface_pressure",
            "100m_u_component_of_wind",
            "100m_v_component_of_wind",
            "10m_wind_gust_since_previous_post_processing",
            "boundary_layer_height",
            "convective_available_potential_energy",
            "k_index",
            "total_column_water_vapour",
            "skin_temperature",
            "mean_convective_precipitation_rate",
            "mean_large_scale_precipitation_rate",
            "mean_snowfall_rate",
            "mean_surface_downward_long_wave_radiation_flux",
            "mean_surface_downward_short_wave_radiation_flux",
            "mean_surface_latent_heat_flux",
            "mean_surface_net_long_wave_radiation_flux",
            "mean_surface_net_short_wave_radiation_flux",
            "mean_surface_sensible_heat_flux",
            "mean_total_precipitation_rate",
            "high_cloud_cover",
            "low_cloud_cover",
            "total_cloud_cover",
            "total_column_cloud_liquid_water",
            "evaporation",
            "runoff",
            "surface_runoff",
            "snow_depth",
            "soil_temperature_level_1",
            "volumetric_soil_water_layer_1",
            "vertically_integrated_moisture_divergence"
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
        'grid:': [1.0, 1.0],
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
    dfm = dfm.drop(columns=["step", "depthBelowLandLayer"])
    dfm['coordinates'] = dfm['latitude'].astype(str) + ',' + dfm['longitude'].astype(str)
    dfm['time'] = pd.to_datetime(dfm['time'], format='%Y-%m-%dT%H:%M:%S.%fZ')
    dfm['day'] = dfm['time'].dt.day
    dfm['month'] = dfm['time'].dt.month
    dfm['year'] = dfm['time'].dt.year
    dfm['hour'] = dfm['time'].dt.hour
    dfm['day of the year'] = dfm['time'].dt.dayofyear
    new_order = ['time', 'day of the year', 'day', 'month', 'year', 'hour', 'latitude', 'longitude', 'coordinates'] + [col for col in dfm.columns if col not in ['time', 'day', 'month', 'year', 'hour', 'latitude', 'longitude', 'coordinates']]
    dfm = dfm[new_order]
    #sst is being dropped because it;s going to be analized in a different pipeline
    dfm = dfm.drop(columns=['sst'])
    #drop date if it comes from different day
    dfm.sort_values(by='time')
    if dfm['day'].iloc[0] != dfm['day'].iloc[-1]:
        dfm.dropna(subset=['t2m'], inplace=True)
    
    # Define log path
    log_path = r"C:\Users\dmoli\Documents\Coding\Weathercast_project\date_log.csv"
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

            # Columns that can tolerate up to 3 full missing days
            relaxed_cols = [
                'swvl1', 'stl1', 'cape', 'tclw', 'sp', 'tcwv', 'sd', 'msl', 'blh', 'tcc',
                'u10', 'v10', 't2m', 'd2m', 'lcc', 'hcc', 'skt', 'u100', 'v100'
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

    # ------- Now: Handle 2-3 hours missing data for specific columns ------- #

    # Columns that need special NaN handling
    target_columns = [
        'cape', 'tclw', 'sp', 'tcwv', 'sd', 'msl', 'blh', 'tcc', 
        'u10', 'v10', 't2m', 'd2m', 'lcc', 'hcc', 'skt', 'u100', 'v100'
    ]


    # Group by coordinate
    for coord, group in dfm.groupby('coordinates'):
        for col in target_columns:
            # Select time and the target variable
            sub_df = group[['time', col]].set_index('time')

            # Find missing values
            nan_mask = sub_df[col].isna()
            if nan_mask.sum() > 0:
                nan_times = sub_df[nan_mask].index

                for t in nan_times:
                    # Define +/- 2 hours window
                    prev_2h = t - pd.Timedelta(hours=2)
                    next_2h = t + pd.Timedelta(hours=2)

                    window = sub_df.loc[prev_2h:next_2h][col]

                    # If 5 expected timestamps and max 2 missing, proceed
                    if len(window) == 5 and window.isna().sum() <= 2:
                        # Check if missing hours are consecutive
                        missing_consec = window.isna().astype(int).diff().abs().sum() == 2

                        if missing_consec:
                            # Interpolate over window
                            dfm.loc[(dfm['coordinates'] == coord) & (dfm['time'].between(prev_2h, next_2h)), col] = \
                                dfm.loc[(dfm['coordinates'] == coord) & (dfm['time'].between(prev_2h, next_2h)), col].interpolate(method='linear', limit_direction='both')
                        else:
                            # Take average of 2 hours before and 2 hours after
                            try:
                                before = sub_df.loc[[prev_2h, t - pd.Timedelta(hours=1)]][col].dropna()
                                after = sub_df.loc[[t + pd.Timedelta(hours=1), next_2h]][col].dropna()
                                if len(before) == 2 and len(after) == 2:
                                    avg_val = pd.concat([before, after]).mean()
                                    dfm.loc[(dfm['coordinates'] == coord) & (dfm['time'] == t), col] = avg_val
                            except Exception as e:
                                print(f"Error filling value for {coord} at {t}:", e)
                                continue
    low_freq_columns = [
        'swvl1', 'stl1', 'surface', 'sro', 'fg10', 'e', 'ro', 'vimd',
        'avg_lsprate', 'avg_cpr', 'avg_tsrwe', 'avg_ishf', 'avg_slhtf',
        'avg_sdswrf', 'avg_sdlwrf', 'avg_snswrf', 'avg_snlwrf', 
        'avg_tprate', 'kx'
    ]

    # Sort first to ensure time continuity
    dfm = dfm.sort_values(['coordinates', 'time'])

    # Interpolate low-frequency columns within each coordinate group
    for col in low_freq_columns:
        dfm[col] = dfm.groupby('coordinates')[col].transform(
            lambda group: group.interpolate(method='linear', limit_direction='both'))
    
    # process information to be stored in the database
    

    dfm['t2m'] = (dfm['t2m'] - 273.15).round(2)
    dfm['d2m'] = (dfm['d2m'] - 273.15).round(2)
    dfm['msl'] = (dfm['msl'] / 100).round(2)
    dfm['sp'] = (dfm['sp'] / 100).round(2)
    dfm['skt'] = (dfm['skt'] - 273.15).round(2)
    dfm['stl1'] = (dfm['stl1'] - 273.15).round(2)
    dfm['kx'] = (dfm['kx'] - 273.15).round(2)

    #identify outliers and store in DB
    # Initialize list to collect outliers
    outliers_list = []

    # Exclude metadata columns
    meta_cols = ['time', 'day of the year', 'day', 'month', 'year', 'hour', 'coordinates']
    data_cols = [col for col in dfm.select_dtypes(include=[np.number]).columns if col not in meta_cols]

    # Group by coordinate
    for coord, group in dfm.groupby('coordinates'):
        for col in data_cols:
            if group[col].isnull().all():
                continue  # skip all-NaN columns

            try:
                # Compute z-scores
                col_data = group[col].dropna()
                if col_data.std() < 1e-8:
                    continue  # skip nearly constant series``
                z_scores = zscore(col_data)
                outlier_mask = np.abs(z_scores) > 3

                # Get times and values
                outlier_values = group.loc[group[col].dropna().index[outlier_mask], ['time', col]]
                for _, row in outlier_values.iterrows():
                    outliers_list.append({
                        'coordinates': coord,
                        'date': row['time'].date(),
                        'time': row['time'].time(),
                        'value': row[col],
                        'variable': col
                    })
            except Exception as e:
                print(f"Error computing outliers for {coord}, {col}: {e}")
                continue

    # Convert to DataFrame
    outliers_df = pd.DataFrame(outliers_list)

    # Save to DB
    
    # Connection details
    user = 'root'
    password = 'Hamilton1186!'
    host = '127.0.0.1'
    port = '3306'
    db = 'weatherdb'

    # Create the engine for MySQL
    engine = create_engine(f'mysql+pymysql://{user}:{password}@{host}:{port}/{db}')

    # Save to MySQL table
    outliers_df.to_sql(
        name='outliers',
        con=engine,
        if_exists='append', 
        index=False
    )
    # Dispose of the engine to close the connection
    engine.dispose()
    # Outlier threshold (as absolute count)
    threshold_per_variable = 15000

    for var in outliers_df['variable'].unique():
        var_df = outliers_df[outliers_df['variable'] == var]

        if len(var_df) > threshold_per_variable:
            # generate and store boxplot
            os.makedirs('outlier_boxplots', exist_ok=True)

            for var in outliers_df['variable'].unique():
                var_df = outliers_df[outliers_df['variable'] == var]
                if len(var_df) > threshold_per_variable:
                    # Sample a subset if needed (e.g., 5000 points for readability)
                    plot_df = var_df.sample(n=min(5000, len(var_df)), random_state=42)
                    plt.figure(figsize=(8, 6))
                    sns.boxplot(y='value', data=plot_df)  # Just show value distribution
                    plt.title(f'Boxplot of Outliers - {var} ({len(var_df)} points)')
                    plt.ylabel(var)
                    plt.tight_layout()
                    plt.savefig(rf'C:\Users\dmoli\Documents\Coding\Weathercast_project\boxplot_{var}.png')
                    plt.close()

    # Save the DataFrame to a CSV file    
    # Group and compute statistics
    #instead of using groupby, we can use the make a list and then execute a loop
    #grouped = dfm.groupby('coordinates')
    #summary_df = grouped.agg({'swvl1': ['mean', 'min', 'max', 'std', 'median', lambda x: x.mode().iloc[0], lambda x: x.quantile(0.25), lambda x: x.quantile(0.75), lambda x: x.max() - x.min(), skew, lambda x: kurtosis(x, fisher=True), np.var],
                            #'stl1': ['mean', 'min', 'max', 'std', 'median', lambda x: x.mode().iloc[0], lambda x: x.quantile(0.25), lambda x: x.quantile(0.75), lambda x: x.max() - x.min(), skew, lambda x: kurtosis(x, fisher=True), np.var],
                            #'surface': ['mean', 'min', 'max', 'std', 'median', lambda x: x.mode().iloc[0], lambda x: x.quantile(0.25), lambda x: x.quantile(0.75), lambda x: x.max() - x.min(), skew, lambda x: kurtosis(x, fisher=True), np.var],
                            #'sro': ['mean', 'min', 'max', 'std', 'median', lambda x: x.mode().iloc[0], lambda x: x.quantile(0.25), lambda x: x.quantile(0.75), lambda x: x.max() - x.min(), skew, lambda x: kurtosis(x, fisher=True)],
                            #'e': ['mean', 'min', 'max', 'std', 'median', lambda x: x.mode().iloc[0], lambda x: x.quantile(0.25), lambda x: x.quantile(0.75), lambda x: x.max() - x.min(), skew, lambda x: kurtosis(x, fisher=True), np.var],
                            #'ro': ['mean', 'min', 'max', 'std', 'median', lambda x: x.mode().iloc[0], lambda x: x.quantile(0.25), lambda x: x.quantile(0.75), lambda x: x.max() - x.min(), skew, lambda x: kurtosis(x, fisher=True), np.var],
                            #'vimd': ['mean', 'min', 'max', 'std', 'median', lambda x: x.mode().iloc[0], lambda x: x.quantile(0.25), lambda x: x.quantile(0.75), lambda x: x.max() - x.min(), skew, lambda x: kurtosis(x, fisher=True), np.var],
                            #'avg_lsprate': ['mean', 'min', 'max', 'std', 'median', lambda x: x.mode().iloc[0], lambda x: x.quantile(0.25), lambda x: x.quantile(0.75), lambda x: x.max() - x.min(), skew, lambda x: kurtosis(x, fisher=True), np.var],
                            #'avg_cpr': ['mean', 'min', 'max', 'std', 'median', lambda x: x.mode().iloc[0], lambda x: x.quantile(0.25), lambda x: x.quantile(0.75), lambda x: x.max() - x.min(), skew, lambda x: kurtosis(x, fisher=True), np.var],
                            #'avg_tsrwe': ['mean', 'min', 'max', 'std', 'median', lambda x: x.mode().iloc[0], lambda x: x.quantile(0.25), lambda x: x.quantile(0.75), lambda x: x.max() - x.min(), skew, lambda x: kurtosis(x, fisher=True), np.var],
                            #'avg_ishf': ['mean', 'min', 'max', 'std', 'median', lambda x: x.mode().iloc[0], lambda x: x.quantile(0.25), lambda x: x.quantile(0.75), lambda x: x.max() - x.min(), skew, lambda x: kurtosis(x, fisher=True), np.var], 
                            #'avg_slhtf': ['mean', 'min', 'max', 'std', 'median', lambda x: x.mode().iloc[0], lambda x: x.quantile(0.25), lambda x: x.quantile(0.75), lambda x: x.max() - x.min(), skew, lambda x: kurtosis(x, fisher=True), np.var],
                            #'avg_sdswrf': ['mean', 'min', 'max', 'std', 'median', lambda x: x.mode().iloc[0], lambda x: x.quantile(0.25), lambda x: x.quantile(0.75), lambda x: x.max() - x.min(), skew, lambda x: kurtosis(x, fisher=True), np.var],
                            #'avg_sdlwrf': ['mean', 'min', 'max', 'std', 'median', lambda x: x.mode().iloc[0], lambda x: x.quantile(0.25), lambda x: x.quantile(0.75), lambda x: x.max() - x.min(), skew, lambda x: kurtosis(x, fisher=True), np.var],
                            #'avg_snswrf': ['mean', 'min', 'max', 'std', 'median', lambda x: x.mode().iloc[0], lambda x: x.quantile(0.25), lambda x: x.quantile(0.75), lambda x: x.max() - x.min(), skew, lambda x: kurtosis(x, fisher=True), np.var],
                            #'avg_snlwrf': ['mean', 'min', 'max', 'std', 'median', lambda x: x.mode().iloc[0], lambda x: x.quantile(0.25), lambda x: x.quantile(0.75), lambda x: x.max() - x.min(), skew, lambda x: kurtosis(x, fisher=True), np.var],
                            #'avg_tprate': ['mean', 'min', 'max', 'std', 'median', lambda x: x.mode().iloc[0], lambda x: x.quantile(0.25), lambda x: x.quantile(0.75), lambda x: x.max() - x.min(), skew, lambda x: kurtosis(x, fisher=True), np.var],
                            #'cape': ['mean', 'min', 'max', 'std', 'median', lambda x: x.mode().iloc[0], lambda x: x.quantile(0.25), lambda x: x.quantile(0.75), lambda x: x.max() - x.min(), skew, lambda x: kurtosis(x, fisher=True), np.var],
                            #'tclw': ['mean', 'min', 'max', 'std', 'median', lambda x: x.mode().iloc[0], lambda x: x.quantile(0.25), lambda x: x.quantile(0.75), lambda x: x.max() - x.min(), skew, lambda x: kurtosis(x, fisher=True), np.var],
                            #'sp': ['mean', 'min', 'max', 'std', 'median', lambda x: x.mode().iloc[0], lambda x: x.quantile(0.25), lambda x: x.quantile(0.75), lambda x: x.max() - x.min(), skew, lambda x: kurtosis(x, fisher=True), np.var],
                            #'tcwv': ['mean', 'min', 'max', 'std', 'median', lambda x: x.mode().iloc[0], lambda x: x.quantile(0.25), lambda x: x.quantile(0.75), lambda x: x.max() - x.min(), skew, lambda x: kurtosis(x, fisher=True), np.var],
                            #'sd': ['mean', 'min', 'max', 'std', 'median', lambda x: x.mode().iloc[0], lambda x: x.quantile(0.25), lambda x: x.quantile(0.75), lambda x: x.max() - x.min(), skew, lambda x: kurtosis(x, fisher=True), np.var],
                            #'msl': ['mean', 'min', 'max', 'std', 'median', lambda x: x.mode().iloc[0], lambda x: x.quantile(0.25), lambda x: x.quantile(0.75), lambda x: x.max() - x.min(), skew, lambda x: kurtosis(x, fisher=True), np.var],
                            #'blh': ['mean', 'min', 'max', 'std', 'median', lambda x: x.mode().iloc[0], lambda x: x.quantile(0.25), lambda x: x.quantile(0.75), lambda x: x.max() - x.min(), skew, lambda x: kurtosis(x, fisher=True), np.var],
                            #'tcc': ['mean', 'min', 'max', 'std', 'median', lambda x: x.mode().iloc[0], lambda x: x.quantile(0.25), lambda x: x.quantile(0.75), lambda x: x.max() - x.min(), skew, lambda x: kurtosis(x, fisher=True), np.var],
                            #'u10': ['mean', 'min', 'max', 'std', 'median', lambda x: x.mode().iloc[0], lambda x: x.quantile(0.25), lambda x: x.quantile(0.75), lambda x: x.max() - x.min(), skew, lambda x: kurtosis(x, fisher=True), np.var],
                            #'v10': ['mean', 'min', 'max', 'std', 'median', lambda x: x.mode().iloc[0], lambda x: x.quantile(0.25), lambda x: x.quantile(0.75), lambda x: x.max() - x.min(), skew, lambda x: kurtosis(x, fisher=True), np.var],
                            #'t2m': ['mean', 'min', 'max', 'std', 'median', lambda x: x.mode().iloc[0], lambda x: x.quantile(0.25), lambda x: x.quantile(0.75), lambda x: x.max() - x.min(), skew, lambda x: kurtosis(x, fisher=True), np.var],
                            #'d2m': ['mean', 'min', 'max', 'std', 'median', lambda x: x.mode().iloc[0], lambda x: x.quantile(0.25), lambda x: x.quantile(0.75), lambda x: x.max() - x.min(), skew, lambda x: kurtosis(x, fisher=True), np.var],
                            #'lcc': ['mean', 'min', 'max', 'std', 'median', lambda x: x.mode().iloc[0], lambda x: x.quantile(0.25), lambda x: x.quantile(0.75), lambda x: x.max() - x.min(), skew, lambda x: kurtosis(x, fisher=True), np.var],
                            #'hcc': ['mean', 'min', 'max', 'std', 'median', lambda x: x.mode().iloc[0], lambda x: x.quantile(0.25), lambda x: x.quantile(0.75), lambda x: x.max() - x.min(), skew, lambda x: kurtosis(x, fisher=True), np.var],
                            #'skt': ['mean', 'min', 'max', 'std', 'median', lambda x: x.mode().iloc[0], lambda x: x.quantile(0.25), lambda x: x.quantile(0.75), lambda x: x.max() - x.min(), skew, lambda x: kurtosis(x, fisher=True), np.var],
                            #'u100': ['mean', 'min', 'max', 'std', 'median', lambda x: x.mode().iloc[0], lambda x: x.quantile(0.25), lambda x: x.quantile(0.75), lambda x: x.max() - x.min(), skew, lambda x: kurtosis(x, fisher=True), np.var],
                            #'v100': ['mean', 'min', 'max', 'std', 'median', lambda x: x.mode().iloc[0], lambda x: x.quantile(0.25), lambda x: x.quantile(0.75), lambda x: x.max() - x.min(), skew, lambda x: kurtosis(x, fisher=True), np.var],
                            #'kx': ['mean', 'min', 'max', 'std', 'median', lambda x: x.mode().iloc[0], lambda x: x.quantile(0.25), lambda x: x.quantile(0.75), lambda x: x.max() - x.min(), skew, lambda x: kurtosis(x, fisher=True), np.var]
                            #}).reset_index()
    #to avoid critical errors for skew and kurtosis, we need to the define a function, updated above
    #store by grid
    weather = [
    'swvl1', 'stl1', 'surface', 'sro', 'e', 'ro', 'vimd', 'avg_lsprate', 'avg_cpr',
    'avg_tsrwe', 'avg_ishf', 'avg_slhtf', 'avg_sdswrf', 'avg_sdlwrf', 'avg_snswrf',
    'avg_snlwrf', 'avg_tprate', 'cape', 'tclw', 'sp', 'tcwv', 'sd', 'msl', 'blh',
    'tcc', 'u10', 'v10', 't2m', 'd2m', 'lcc', 'hcc', 'skt', 'u100', 'v100', 'kx']

    
    
    common_stats = {
        'mean': 'mean',
        'min': 'min',
        'max': 'max',
        'std': 'std',
        'median': 'median',
        'mode': lambda x: x.mode().iloc[0],
        'q25': lambda x: x.quantile(0.25),
        'q75': lambda x: x.quantile(0.75),
        'range': lambda x: x.max() - x.min(),
        'skew': safe_skew,
        'kurtosis': safe_kurtosis,
        'var': np.var
    }
    agg_dict = {col: [(f'{col}_{key}', func) for key, func in common_stats.items()] for col in weather}
    summary_df = dfm.groupby('coordinates').agg(agg_dict).reset_index()
    summary_df.columns = [col[0] if col[0] == 'coordinates' else col[1] for col in summary_df.columns]
    summary_df['Date'] = dfm['time'].iloc[0]
    summary_df['day of the year'] = pd.to_datetime(summary_df['Date']).dt.dayofyear
    # Create a mapping of coordinates to latitude
    coord_to_lat = dfm[['coordinates', 'latitude']].drop_duplicates().set_index('coordinates')['latitude']
    summary_df['latitude'] = summary_df['coordinates'].map(coord_to_lat)
    day_of_year = summary_df['day of the year'].iloc[0]  
    year = pd.to_datetime(summary_df['Date']).dt.year.iloc[0]  
    base_season = get_season(day_of_year, year)  # Northern Hemisphere season
    season_map = {'Spring': 'Autumn', 'Summer': 'Winter', 'Autumn': 'Spring', 'Winter': 'Summer'}
    summary_df['season'] = summary_df['latitude'].apply(
        lambda lat: season_map[base_season] if lat < 0 else base_season
    )
    #store by region
    weather_reg = [
    'stt', 'sro']
    agg_dict_1 = {col: [(f'{col}_{key}', func) for key, func in common_stats.items()] for col in weather_reg}
    region_mask = xr.open_dataarray("region_mask_global.nc")
    dfm['region'] = dfm.apply(lambda row: get_region_from_mask(row['latitude'], row['longitude'], region_mask), axis=1)
    summary_region = dfm.groupby('region').agg(agg_dict_1).reset_index()
    summary_region.columns = [col[0] if col[0] == 'region' else col[1] for col in summary_region.columns]
    summary_region['Date'] = dfm['time'].iloc[0]
    summary_region['day of the year'] = pd.to_datetime(summary_region['Date']).dt.dayofyear
    # Calculate season for each region
    region_latitudes = dfm.groupby('region')['latitude'].mean()
    summary_region['avg_latitude'] = summary_region['region'].map(region_latitudes)
    base_season = get_season(day_of_year, year)  # Northern Hemisphere season
    season_map = {'Spring': 'Autumn', 'Summer': 'Winter', 'Autumn': 'Spring', 'Winter': 'Summer'}
    summary_region['season'] = summary_region['avg_latitude'].apply(
        lambda lat: season_map[base_season] if lat < 0 else base_season
    )
    summary_region = summary_region.drop(columns=['avg_latitude'])

    #global summary
    weather_glob = [
        'swvl1', 'stl1', 'surface', 'sro', 'e', 'ro', 'vimd', 'avg_lsprate', 'avg_cpr',
        'avg_tsrwe', 'avg_ishf', 'avg_slhtf', 'avg_sdswrf', 'avg_sdlwrf', 'avg_snswrf',
        'avg_snlwrf', 'avg_tprate', 'cape', 'tclw', 'sp', 'tcwv', 'sd', 'msl', 'blh',
        'tcc', 'u10', 'v10', 't2m', 'd2m', 'lcc', 'hcc', 'skt', 'u100', 'v100'
    ]
    agg_dict_global = {col: [(f'{col}_{key}', func) for key, func in common_stats.items()] for col in weather_glob}
    
    # Save to DB
    # MySQL Connection
    user = 'root'
    password = 'Hamilton1186!'
    host = '127.0.0.1'
    port = '3306'
    db = 'weatherdb'
    table_name = 'weather_summary'

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

def get_all_dates(start_date="1940-01-05", end_date="1943-12-31"):
    if end_date is None:
        end_date = datetime.today().strftime('%Y-%m-%d')
    start = datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.strptime(end_date, '%Y-%m-%d')
    delta = end - start
    return [(d.year, d.month, d.day) for d in (start + timedelta(days=i) for i in range(delta.days + 1))]

if __name__ == "__main__":
    all_dates = get_all_dates()
    pool_size = 4  # Number of parallel processes

    with multiprocessing.Pool(pool_size) as pool:
        pool.map(process_single_day, all_dates)

 