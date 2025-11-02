import pandas as pd
import numpy as np
from sqlalchemy import create_engine, inspect
from sqlalchemy.types import String, Float, Integer, DateTime, Date, BigInteger

# --- Define custom aggregation functions ---
# These functions handle potential issues like empty series for mode,
# or insufficient data points for skew/kurtosis.
def safe_skew(x):
    """Calculates skewness, returns NaN if not enough data points."""
    if len(x) > 2: # Skewness requires at least 3 data points
        return x.skew()
    return np.nan

def safe_kurtosis(x):
    """Calculates kurtosis, returns NaN if not enough data points."""
    if len(x) > 3: # Kurtosis requires at least 4 data points
        return x.kurtosis()
    return np.nan

# --- Dummy DataFrame for demonstration ---
# This DataFrame simulates your 'dfm' with all the 'weather' columns
# and some sample data.
data = {
    'coordinates': ['A', 'A', 'B', 'B', 'A', 'C'],
    'time': pd.to_datetime(['2023-01-01', '2023-01-01', '2023-01-01', '2023-01-01', '2023-01-01', '2023-01-01']),
    'swvl1': [10.1, 12.2, 15.3, 11.4, 13.5, 14.6],
    'stl1': [20.1, 21.2, 19.3, 22.4, 20.5, 23.6],
    'surface': [1.0,2.0,3.0,4.0,5.0,6.0],
    'sro': [1.0,1.0,2.0,2.0,3.0,4.0],
    'e': [0.1,0.2,0.1,0.3,0.2,0.4],
    'ro': [0.01,0.02,0.03,0.01,0.02,0.04],
    'vimd': [5.0,6.0,7.0,5.0,8.0,9.0],
    'avg_lsprate': [0.1,0.1,0.2,0.1,0.1,0.3],
    'avg_cpr': [0.5,0.6,0.7,0.5,0.6,0.8],
    'avg_tsrwe': [100.0,101.0,102.0,100.0,101.0,103.0],
    'avg_ishf': [10.0,11.0,12.0,10.0,11.0,13.0],
    'avg_slhtf': [50.0,51.0,52.0,50.0,51.0,53.0],
    'avg_sdswrf': [1000.0,1001.0,1002.0,1000.0,1001.0,1003.0],
    'avg_sdlwrf': [500.0,501.0,502.0,500.0,501.0,503.0],
    'avg_snswrf': [10.0,11.0,12.0,10.0,11.0,13.0],
    'avg_snlwrf': [5.0,6.0,7.0,5.0,6.0,8.0],
    'avg_tprate': [0.01,0.02,0.01,0.02,0.01,0.03],
    'cape': [100.0,120.0,110.0,130.0,105.0,140.0],
    'tclw': [0.1,0.2,0.1,0.2,0.1,0.3],
    'sp': [100000.0,100001.0,100002.0,100000.0,100001.0,100003.0],
    'tcwv': [20.0,21.0,22.0,20.0,21.0,23.0],
    'sd': [0.1,0.1,0.2,0.1,0.1,0.3],
    'msl': [101000.0,101001.0,101002.0,101000.0,101001.0,101003.0],
    'blh': [1000.0,1100.0,1050.0,1150.0,1020.0,1200.0],
    'tcc': [0.5,0.6,0.5,0.6,0.5,0.7],
    'u10': [5.0,6.0,5.0,6.0,5.0,7.0],
    'v10': [2.0,3.0,2.0,3.0,2.0,4.0],
    't2m': [280.0,281.0,282.0,280.0,281.0,283.0],
    'd2m': [270.0,271.0,272.0,270.0,271.0,273.0],
    'lcc': [0.1,0.2,0.1,0.2,0.1,0.3],
    'hcc': [0.05,0.06,0.05,0.06,0.05,0.07],
    'skt': [285.0,286.0,287.0,285.0,286.0,288.0],
    'u100': [10.0,11.0,10.0,11.0,10.0,12.0],
    'v100': [4.0,5.0,4.0,5.0,4.0,6.0],
    'kx': [10.0,11.0,12.0,10.0,11.0,13.0]
}
dfm = pd.DataFrame(data)

# Columns to aggregate
weather = [
    'swvl1', 'stl1', 'surface', 'sro', 'e', 'ro', 'vimd', 'avg_lsprate', 'avg_cpr',
    'avg_tsrwe', 'avg_ishf', 'avg_slhtf', 'avg_sdswrf', 'avg_sdlwrf', 'avg_snswrf',
    'avg_snlwrf', 'avg_tprate', 'cape', 'tclw', 'sp', 'tcwv', 'sd', 'msl', 'blh',
    'tcc', 'u10', 'v10', 't2m', 'd2m', 'lcc', 'hcc', 'skt', 'u100', 'v100', 'kx'
]

# Common stats
common_stats = {
    'mean': 'mean',
    'min': 'min',
    'max': 'max',
    'std': 'std',
    'median': 'median',
    'mode': lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan,
    'q25': lambda x: x.quantile(0.25),
    'q75': lambda x: x.quantile(0.75),
    'range': lambda x: x.max() - x.min(),
    'skew': safe_skew,
    'kurtosis': safe_kurtosis,
    'var': np.var
}

# Apply aggregation
agg_dict = {col: [(f'{col}_{key}', func) for key, func in common_stats.items()] for col in weather}
summary_df = dfm.groupby('coordinates').agg(agg_dict).reset_index()
summary_df.columns = [col[0] if col[0] == 'coordinates' else col[1] for col in summary_df.columns]
summary_df['Date'] = dfm['time'].iloc[0]
summary_df['day_of_year'] = pd.to_datetime(summary_df['Date']).dt.dayofyear

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
