import cdsapi
import cfgrib
import pandas as pd
import os

# Create lists for years and months (as strings for CDS API)
years = [str(year) for year in range(1940, 2024)]
months = [f"{month:02d}" for month in range(1, 13)]  # Two-digit strings: "01", "02", ..., "12"

# Initialize CDS API client
client = cdsapi.Client()

for i in years:
    for j in months:
        # Set days based on month and leap year
        if j == "02":  # February
            if int(i) % 4 == 0 and (int(i) % 100 != 0 or int(i) % 400 == 0):  # leap year check
                days = [f"{day:02d}" for day in range(1, 30)]  # 29 days for leap year
            else:
                days = [f"{day:02d}" for day in range(1, 29)]  # 28 days for non-leap year
        elif j in ["04", "06", "09", "11"]:
            days = [f"{day:02d}" for day in range(1, 31)]  # 30 days
        else:  # January, March, May, July, August, October, December
            days = [f"{day:02d}" for day in range(1, 32)]  # 31 days
        #API request for each day in the month for every year
        for k in days:
            try:
                # Define the GRIB file name
                grib_file = f"{i}_{j}_{k}_era5.grib"

                # CDS API request
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
                    "year": [i],  # Use loop variable
                    "month": [j],
                    "day": [k],
                    "time": [
                        "00:00", "01:00", "02:00", "03:00", "04:00", "05:00",
                        "06:00", "07:00", "08:00", "09:00", "10:00", "11:00",
                        "12:00", "13:00", "14:00", "15:00", "16:00", "17:00",
                        "18:00", "19:00", "20:00", "21:00", "22:00", "23:00"
                    ],
                    "data_format": "grib",
                    "download_format": "unarchived",
                    "area": [90, -180, -90, 180],
                    "grid": [1.0, 1.0]
                }

                # Download GRIB file
                print(f"Downloading {grib_file}...")
                client.retrieve(dataset, request).download(grib_file)

                # Open GRIB file
                ds = cfgrib.open_datasets(grib_file)

                # Initialize a list to store DataFrames
                dfs = []
                for dataset in ds:
                    df = dataset.to_dataframe().reset_index()
                    dfs.append(df)

                # Merge DataFrames
                if len(dfs) > 1:
                    merged_df = dfs[0]
                    for df in dfs[1:]:
                        merged_df = merged_df.merge(df, on=['time', 'latitude', 'longitude'], how='outer')
                else:
                    merged_df = dfs[0]

                # Select relevant columns
                variables = ['stl1', 'siconc', 'sst', 'sp', 'tcc', 't2m', 'd2m', 'u10n', 'u100']
                output_columns = ['time', 'latitude', 'longitude'] + [col for col in merged_df.columns if col in variables]
                merged_df = merged_df[output_columns]

                # Convert units and add coordinates
                merged_df['t2m_celsius'] = merged_df['t2m'] - 273.15
                merged_df['dewpoint_temperature_celsius'] = merged_df['d2m'] - 273.15
                merged_df['sea_surface_temperature_celsius'] = merged_df['sst'] - 273.15
                merged_df['surface_pressure_hPa'] = merged_df['sp'] * 0.01
                merged_df['surface_temperature_level_1_cel'] = merged_df['stl1'] - 273.15
                merged_df['soil_temperature_level_1'] = merged_df['stl1'] - 273.15
                merged_df['coordinates'] = merged_df['latitude'].astype(str) + ',' + merged_df['longitude'].astype(str)


                # consider the following for deeper analysis
                # 't2m_celsius': [..., lambda x: (x > 30).sum()]  # Count temperatures > 30°C
                #'siconc': [..., lambda x: (x < 0.1).sum()]      # Count low sea ice concentrations
                #Correlation with Another Variable:
                #Compute the correlation between two columns within each group (e.g., temperature vs. pressure).

                #Example:
                #python
                # find the way to get your correlation matrix daily. making a column for the highest correlation  and the name
                #'t2m_celsius': [..., lambda x: x.corr(merged_df.loc[x.index, 'surface_pressure_hPa'])]


                # Group and compute statistics
                grouped = merged_df.groupby('coordinates')
                summary_df = grouped.agg({
                    't2m_celsius': ['mean', 'min', 'max', 'std', 'median', lambda x: x.mode().iloc[0], lambda x: x.quantile(0.25), lambda x: x.quantile(0.75), lambda x: x.max() - x.min(), 'skew', 'kurt', 'var'],
                    'dewpoint_temperature_celsius': ['mean', 'min', 'max', 'std', 'median', lambda x: x.mode().iloc[0], lambda x: x.quantile(0.25), lambda x: x.quantile(0.75), lambda x: x.max() - x.min(), 'skew', 'kurt', 'var'],
                    'sea_surface_temperature_celsius': ['mean', 'min', 'max', 'std', 'median', lambda x: x.mode().iloc[0], lambda x: x.quantile(0.25), lambda x: x.quantile(0.75), lambda x: x.max() - x.min(), 'skew', 'kurt', 'var'],
                    'surface_pressure_hPa': ['mean', 'min', 'max', 'std', 'median', lambda x: x.mode().iloc[0], lambda x: x.quantile(0.25), lambda x: x.quantile(0.75), lambda x: x.max() - x.min(), 'skew', 'kurt'],
                    'surface_temperature_level_1_cel': ['mean', 'min', 'max', 'std', 'median', lambda x: x.mode().iloc[0], lambda x: x.quantile(0.25), lambda x: x.quantile(0.75), lambda x: x.max() - x.min(), 'skew', 'kurt', 'var'],
                    'soil_temperature_level_1': ['mean', 'min', 'max', 'std', 'median', lambda x: x.mode().iloc[0], lambda x: x.quantile(0.25), lambda x: x.quantile(0.75), lambda x: x.max() - x.min(), 'skew', 'kurt', 'var'],
                    'u100': ['mean', 'min', 'max', 'std', 'median', lambda x: x.mode().iloc[0], lambda x: x.quantile(0.25), lambda x: x.quantile(0.75), lambda x: x.max() - x.min(), 'skew', 'kurt', 'var'],
                    'u10n': ['mean', 'min', 'max', 'std', 'median', lambda x: x.mode().iloc[0], lambda x: x.quantile(0.25), lambda x: x.quantile(0.75), lambda x: x.max() - x.min(), 'skew', 'kurt', 'var'],
                    'tcc': ['mean', 'min', 'max', 'std', 'median', lambda x: x.mode().iloc[0], lambda x: x.quantile(0.25), lambda x: x.quantile(0.75), lambda x: x.max() - x.min(), 'skew', 'kurt', 'var'],
                    'siconc': ['mean', 'min', 'max', 'std', 'median', lambda x: x.mode().iloc[0], lambda x: x.quantile(0.25), lambda x: x.quantile(0.75), lambda x: x.max() - x.min(), 'skew', 'kurt', 'var'] 
                }).reset_index()

                summary_df.columns = ['_'.join(col).rstrip('_') for col in summary_df.columns.values]
                summary_df['Data'] = df['time'].iloc[0]

                # Append to CSV
                file_exists = os.path.exists('era5_summary_by_coordinate.csv')
                summary_df.to_csv('era5_summary_by_coordinate.csv', 
                                  mode='a', 
                                  header=not file_exists, 
                                  index=False)
                print(f"Data for {i}-{j}-{k} appended to era5_summary_by_coordinate.csv")

            except Exception as e:
                print(f"Error processing {i}-{j}-{k}: {e}")
            finally:
                # Delete GRIB file if it exists
                if os.path.exists(grib_file):
                    try:
                        os.remove(grib_file)
                        print(f"Deleted {grib_file}")
                    except Exception as e:
                        print(f"Error deleting {grib_file}: {e}")