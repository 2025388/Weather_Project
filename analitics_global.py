import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy import create_engine

# MySQL Connection
user = 'root'
password = 'Hamilton1186!'
host = '127.0.0.1'
port = '3306'
db = 'weatherdb'
table_name = 'weather_summary_global'

# Create SQLAlchemy engine
engine = create_engine(f"mysql+pymysql://{user}:{password}@{host}:{port}/{db}")

df = pd.read_sql_table(table_name, engine)

fig, ax1 = plt.subplots()
# graph data    

plt.title('Global Weather Summary')
sns.lineplot(data=df, x='Date', y='t2m_weighted_mean',  label='Average Temperature', color='blue', ax=ax1)
sns.lineplot(data=df, x='Date', y='t2m_min',  label='min Temperature', color='red', ax=ax1)
sns.lineplot(data=df, x='Date', y='t2m_max',  label='max Temperature', color='red', ax=ax1)
sns.lineplot(data=df, x='Date', y='t2m_median',  label='median Temperature', color='green', ax=ax1)
sns.lineplot(data=df, x='Date', y='d2m_weighted_mean',  label='due point Temperature', color='yellow', ax=ax1)
ax2 = ax1.twinx()
sns.lineplot(data=df, x='Date', y='tcc_weighted_mean',  label='cloud coverage', color='orange', ax=ax2)
plt.xlabel('Date')


plt.show()