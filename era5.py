import cdsapi
import cfgrib
import pandas as pd
import xarray as xr
import os as os


# create list of all the ranges we want to download
years = [str(year) for year in range(1940, 2024)]
months = [f"{month:02d}" for month in range(1, 13)]  # Two-digit strings: "01", "02", ..., "12"



file_exists = os.path.exists('era5_summary_by_coordinate.csv')

for i in years:
    for j in months:
        if j == "02": #check if leap year
            if int(i) % 4 == 0:
                days = [day for day in range(1, 30)]
                for k in days:
                    dataset = "reanalysis-era5-single-levels"
                    request = {
                        "product_type": ["reanalysis"],
                        "variable": [
                            "2m_dewpoint_temperature",
                            "2m_temperature",
                            "sea_surface_temperature",
                            "surface_pressure",
                            "100m_u_component_of_wind",
                            "10m_u_component_of_neutral_wind",
                            "total_cloud_cover",
                            "soil_temperature_level_1",
                            "sea_ice_cover"
                        ],
                        "year": ["i"],
                        "month": ["j"],
                        "day": ["k"],
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
                        "grid": [1.0, 1.0]  # 1x1 degree resolution
                    }

                    client = cdsapi.Client()
                    client.retrieve(dataset, request).download('i','j', 'k', 'era5.grib')
                    ds = cfgrib.open_datasets("'i','j', 'k', 'era5.grib'")
                    # Initialize a list to store DataFrames
                    dfs = []

                    # Process each dataset
                    for dataset in ds:
                        # Convert dataset to DataFrame and reset index to include coordinates
                        df = dataset.to_dataframe().reset_index()
                        dfs.append(df)

                    # Merge DataFrames on common coordinates (time, latitude, longitude)
                    if len(dfs) > 1:
                        merged_df = dfs[0]
                        for df in dfs[1:]:
                            merged_df = merged_df.merge(df, on=['time', 'latitude', 'longitude'], how='outer')
                    else:
                        merged_df = dfs[0]

                    # Select relevant columns (coordinates + variables)
                    variables = ['stl1', 'siconc', 'sst', 'sp', 'tcc', 't2m', 'd2m', 'u10n', 'u100', ]
                    output_columns = ['time', 'latitude', 'longitude'] + [col for col in merged_df.columns if col in variables]
                    merged_df = merged_df[output_columns]

                    merged_df = merged_df.reset_index()

                    merged_df ['t2m_celsius'] = merged_df['t2m'] - 273.15
                    merged_df ['dewpoint_temperature_celsius'] = merged_df['d2m'] - 273.15
                    merged_df ['sea_surface_temperature_celsius'] = merged_df['sst'] - 273.15
                    merged_df ['surface_pressure_hPa'] = merged_df['sp'] * 0.01
                    merged_df ['surface_temperature_level_1_cel'] = merged_df['stl1'] - 273.15
                    merged_df ['soil_temperature_level_1']  = merged_df['stl1'] - 273.15
                    merged_df ['coordinates'] = merged_df['latitude'].astype(str) + ',' + merged_df['longitude'].astype(str)
                    grouped = merged_df.groupby('coordinates')
                    summary_df = grouped.agg({
                        't2m_celsius': ['mean', 'min', 'max', 'std', 'median'],
                        'dewpoint_temperature_celsius': ['mean', 'min', 'max', 'std', 'median'],
                        'sea_surface_temperature_celsius': ['mean', 'min', 'max', 'std', 'median'],
                        'surface_pressure_hPa': ['mean', 'min', 'max', 'std', 'median'],
                        'surface_temperature_level_1_cel': ['mean', 'min', 'max', 'std', 'median'],
                        'soil_temperature_level_1': ['mean', 'min', 'max', 'std', 'median'],
                        'u100': ['mean', 'min', 'max', 'std', 'median'],
                        'u10n': ['mean', 'min', 'max', 'std', 'median'],
                        'tcc': ['mean', 'min', 'max', 'std', 'median'],
                        'siconc': ['mean', 'min', 'max', 'std', 'median']
                    }).reset_index()

                    summary_df.columns = ['_'.join(col).rstrip('_') for col in summary_df.columns.values]
                    summary_df ['Data'] = df['time'].iloc[0]
                    summary_df.to_csv('era5_summary_by_coordinate.csv', mode='a', header=not file_exists,  index=False)
                    
            else:
                days = [day for day in range(1, 29)]
                for k in days:
                    dataset = "reanalysis-era5-single-levels"
                    request = {
                        "product_type": ["reanalysis"],
                        "variable": [
                            "2m_dewpoint_temperature",
                            "2m_temperature",
                            "sea_surface_temperature",
                            "surface_pressure",
                            "100m_u_component_of_wind",
                            "10m_u_component_of_neutral_wind",
                            "total_cloud_cover",
                            "soil_temperature_level_1",
                            "sea_ice_cover"
                        ],
                        "year": ["i"],
                        "month": ["j"],
                        "day": ["k"],
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
                        "grid": [1.0, 1.0]  # 1x1 degree resolution
                    }

                    client = cdsapi.Client()
                    client.retrieve(dataset, request).download('i','j', 'k', 'era5.grib')
                    ds = cfgrib.open_datasets("'i','j', 'k', 'era5.grib'")
                    # Initialize a list to store DataFrames
                    dfs = []

                    # Process each dataset
                    for dataset in ds:
                        # Convert dataset to DataFrame and reset index to include coordinates
                        df = dataset.to_dataframe().reset_index()
                        dfs.append(df)

                    # Merge DataFrames on common coordinates (time, latitude, longitude)
                    if len(dfs) > 1:
                        merged_df = dfs[0]
                        for df in dfs[1:]:
                            merged_df = merged_df.merge(df, on=['time', 'latitude', 'longitude'], how='outer')
                    else:
                        merged_df = dfs[0]

                    # Select relevant columns (coordinates + variables)
                    variables = ['stl1', 'siconc', 'sst', 'sp', 'tcc', 't2m', 'd2m', 'u10n', 'u100', ]
                    output_columns = ['time', 'latitude', 'longitude'] + [col for col in merged_df.columns if col in variables]
                    merged_df = merged_df[output_columns]

                    merged_df = merged_df.reset_index()

                    merged_df ['t2m_celsius'] = merged_df['t2m'] - 273.15
                    merged_df ['dewpoint_temperature_celsius'] = merged_df['d2m'] - 273.15
                    merged_df ['sea_surface_temperature_celsius'] = merged_df['sst'] - 273.15
                    merged_df ['surface_pressure_hPa'] = merged_df['sp'] * 0.01
                    merged_df ['surface_temperature_level_1_cel'] = merged_df['stl1'] - 273.15
                    merged_df ['soil_temperature_level_1']  = merged_df['stl1'] - 273.15
                    merged_df ['coordinates'] = merged_df['latitude'].astype(str) + ',' + merged_df['longitude'].astype(str)
                    grouped = merged_df.groupby('coordinates')
                    summary_df = grouped.agg({
                        't2m_celsius': ['mean', 'min', 'max', 'std', 'median'],
                        'dewpoint_temperature_celsius': ['mean', 'min', 'max', 'std', 'median'],
                        'sea_surface_temperature_celsius': ['mean', 'min', 'max', 'std', 'median'],
                        'surface_pressure_hPa': ['mean', 'min', 'max', 'std', 'median'],
                        'surface_temperature_level_1_cel': ['mean', 'min', 'max', 'std', 'median'],
                        'soil_temperature_level_1': ['mean', 'min', 'max', 'std', 'median'],
                        'u100': ['mean', 'min', 'max', 'std', 'median'],
                        'u10n': ['mean', 'min', 'max', 'std', 'median'],
                        'tcc': ['mean', 'min', 'max', 'std', 'median'],
                        'siconc': ['mean', 'min', 'max', 'std', 'median']
                    }).reset_index()

                    summary_df.columns = ['_'.join(col).rstrip('_') for col in summary_df.columns.values]
                    summary_df ['Data'] = df['time'].iloc[0]
                    summary_df.to_csv('era5_summary_by_coordinate.csv', mode='a', header=not file_exists,  index=False)
                    
        elif j in ["04", "06", "09", "11"]:
            days = [day for day in range(1, 31)]
            for k in days:
                dataset = "reanalysis-era5-single-levels"
                request = {
                        "product_type": ["reanalysis"],
                        "variable": [
                            "2m_dewpoint_temperature",
                            "2m_temperature",
                            "sea_surface_temperature",
                            "surface_pressure",
                            "100m_u_component_of_wind",
                            "10m_u_component_of_neutral_wind",
                            "total_cloud_cover",
                            "soil_temperature_level_1",
                            "sea_ice_cover"
                        ],
                        "year": ["i"],
                        "month": ["j"],
                        "day": ["k"],
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
                        "grid": [1.0, 1.0]  }
                client = cdsapi.Client()
                client.retrieve(dataset, request).download('i','j', 'k', 'era5.grib')
                ds = cfgrib.open_datasets("'i','j', 'k', 'era5.grib'")
                # Initialize a list to store DataFrames
                dfs = []
                # Process each dataset
                for dataset in ds:
                    # Convert dataset to DataFrame and reset index to include coordinates
                    df = dataset.to_dataframe().reset_index()
                    dfs.append(df)
                    # Merge DataFrames on common coordinates (time, latitude, longitude)
                    if len(dfs) > 1:
                        merged_df = dfs[0]
                        for df in dfs[1:]:
                            merged_df = merged_df.merge(df, on=['time', 'latitude', 'longitude'], how='outer')
                    else:
                        merged_df = dfs[0]

                    # Select relevant columns (coordinates + variables)
                    variables = ['stl1', 'siconc', 'sst', 'sp', 'tcc', 't2m', 'd2m', 'u10n', 'u100', ]
                    output_columns = ['time', 'latitude', 'longitude'] + [col for col in merged_df.columns if col in variables]
                    merged_df = merged_df[output_columns]

                    merged_df = merged_df.reset_index()

                    merged_df ['t2m_celsius'] = merged_df['t2m'] - 273.15
                    merged_df ['dewpoint_temperature_celsius'] = merged_df['d2m'] - 273.15
                    merged_df ['sea_surface_temperature_celsius'] = merged_df['sst'] - 273.15
                    merged_df ['surface_pressure_hPa'] = merged_df['sp'] * 0.01
                    merged_df ['surface_temperature_level_1_cel'] = merged_df['stl1'] - 273.15
                    merged_df ['soil_temperature_level_1']  = merged_df['stl1'] - 273.15
                    merged_df ['coordinates'] = merged_df['latitude'].astype(str) + ',' + merged_df['longitude'].astype(str)
                    grouped = merged_df.groupby('coordinates')
                    summary_df = grouped.agg({
                        't2m_celsius': ['mean', 'min', 'max', 'std', 'median'],
                        'dewpoint_temperature_celsius': ['mean', 'min', 'max', 'std', 'median'],
                        'sea_surface_temperature_celsius': ['mean', 'min', 'max', 'std', 'median'],
                        'surface_pressure_hPa': ['mean', 'min', 'max', 'std', 'median'],
                        'surface_temperature_level_1_cel': ['mean', 'min', 'max', 'std', 'median'],
                        'soil_temperature_level_1': ['mean', 'min', 'max', 'std', 'median'],
                        'u100': ['mean', 'min', 'max', 'std', 'median'],
                        'u10n': ['mean', 'min', 'max', 'std', 'median'],
                        'tcc': ['mean', 'min', 'max', 'std', 'median'],
                        'siconc': ['mean', 'min', 'max', 'std', 'median']
                    }).reset_index()

                    summary_df.columns = ['_'.join(col).rstrip('_') for col in summary_df.columns.values]
                    summary_df ['Data'] = df['time'].iloc[0]
                    summary_df.to_csv('era5_summary_by_coordinate.csv', mode='a', header=not file_exists,  index=False)
                
        elif j in ["01", "03", "05", "07", "08", "10", "12"]:
            days = [day for day in range(1, 32)]
            for k in days:
                dataset = "reanalysis-era5-single-levels"
                request = {
                    "product_type": ["reanalysis"],
                    "variable": [
                            "2m_dewpoint_temperature",
                            "2m_temperature",
                            "sea_surface_temperature",
                            "surface_pressure",
                            "100m_u_component_of_wind",
                            "10m_u_component_of_neutral_wind",
                            "total_cloud_cover",
                            "soil_temperature_level_1",
                            "sea_ice_cover"
                        ],
                        "year": ["i"],
                        "month": ["j"],
                        "day": ["k"],
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
                        "grid": [1.0, 1.0]}

                client = cdsapi.Client()
                client.retrieve(dataset, request).download('i','j', 'k', 'era5.grib')
                ds = cfgrib.open_datasets("'i','j', 'k', 'era5.grib'")
                # Initialize a list to store DataFrames
                dfs = []
                # Process each dataset
                for dataset in ds:
                    # Convert dataset to DataFrame and reset index to include coordinates
                    df = dataset.to_dataframe().reset_index()
                    dfs.append(df)
                    # Merge DataFrames on common coordinates (time, latitude, longitude)
                    if len(dfs) > 1:
                        merged_df = dfs[0]
                        for df in dfs[1:]:
                            merged_df = merged_df.merge(df, on=['time', 'latitude', 'longitude'], how='outer')
                    else:
                        merged_df = dfs[0]

                    # Select relevant columns (coordinates + variables)
                    variables = ['stl1', 'siconc', 'sst', 'sp', 'tcc', 't2m', 'd2m', 'u10n', 'u100', ]
                    output_columns = ['time', 'latitude', 'longitude'] + [col for col in merged_df.columns if col in variables]
                    merged_df = merged_df[output_columns]

                    merged_df = merged_df.reset_index()

                    merged_df ['t2m_celsius'] = merged_df['t2m'] - 273.15
                    merged_df ['dewpoint_temperature_celsius'] = merged_df['d2m'] - 273.15
                    merged_df ['sea_surface_temperature_celsius'] = merged_df['sst'] - 273.15
                    merged_df ['surface_pressure_hPa'] = merged_df['sp'] * 0.01
                    merged_df ['surface_temperature_level_1_cel'] = merged_df['stl1'] - 273.15
                    merged_df ['soil_temperature_level_1']  = merged_df['stl1'] - 273.15
                    merged_df ['coordinates'] = merged_df['latitude'].astype(str) + ',' + merged_df['longitude'].astype(str)
                    grouped = merged_df.groupby('coordinates')
                    summary_df = grouped.agg({
                        't2m_celsius': ['mean', 'min', 'max', 'std', 'median'],
                        'dewpoint_temperature_celsius': ['mean', 'min', 'max', 'std', 'median'],
                        'sea_surface_temperature_celsius': ['mean', 'min', 'max', 'std', 'median'],
                        'surface_pressure_hPa': ['mean', 'min', 'max', 'std', 'median'],
                        'surface_temperature_level_1_cel': ['mean', 'min', 'max', 'std', 'median'],
                        'soil_temperature_level_1': ['mean', 'min', 'max', 'std', 'median'],
                        'u100': ['mean', 'min', 'max', 'std', 'median'],
                        'u10n': ['mean', 'min', 'max', 'std', 'median'],
                        'tcc': ['mean', 'min', 'max', 'std', 'median'],
                        'siconc': ['mean', 'min', 'max', 'std', 'median']
                    }).reset_index()

                    summary_df.columns = ['_'.join(col).rstrip('_') for col in summary_df.columns.values]
                    summary_df ['Data'] = df['time'].iloc[0]
                    summary_df.to_csv('era5_summary_by_coordinate.csv', mode='a', header=not file_exists,  index=False)
                