import numpy as np
import xarray as xr


# Step 1: Define 1x1 degree global grid
lats = np.arange(-90, 91, 1.0)
lons = np.arange(-180, 181, 1.0)

region_mask = xr.DataArray(
    data=np.full((len(lats), len(lons)), fill_value='Other', dtype=object),
    coords={"lat": lats, "lon": lons},
    dims=["lat", "lon"]
)

# Step 2: Define main regions and refined subregions using bounding boxes
regions = [
    # North America & USA Subregions
    {"name": "Pacific Northwest", "lat": (42, 50), "lon": (-125, -115)},
    {"name": "California", "lat": (32, 42), "lon": (-125, -114)},
    {"name": "Southwest USA", "lat": (30, 40), "lon": (-115, -100)},
    {"name": "Great Plains", "lat": (35, 49), "lon": (-105, -90)},
    {"name": "Midwest USA", "lat": (36, 47), "lon": (-95, -80)},
    {"name": "Northeast USA", "lat": (38, 47), "lon": (-80, -67)},
    {"name": "Southeast USA", "lat": (25, 38), "lon": (-90, -75)},
    {"name": "Florida", "lat": (24, 31), "lon": (-87.5, -80)},

    # Canada Subregions
    {"name": "Western Canada", "lat": (49, 70), "lon": (-140, -110)},
    {"name": "Central Canada", "lat": (49, 70), "lon": (-110, -85)},
    {"name": "Eastern Canada", "lat": (45, 70), "lon": (-85, -50)},
    {"name": "Northern Canada & Arctic Archipelago", "lat": (65, 83), "lon": (-140, -50)},

    {"name": "Mexico", "lat": (15, 25), "lon": (-120, -85)},
    {"name": "Central America", "lat": (5, 15), "lon": (-95, -75)},
    {"name": "Caribbean", "lat": (10, 25), "lon": (-85, -60)},

    # South America
    {"name": "Amazon Basin", "lat": (-10, 5), "lon": (-75, -50)},
    {"name": "Andes Mountains", "lat": (-20, 5), "lon": (-80, -60)},
    {"name": "Southern Cone", "lat": (-60, -20), "lon": (-75, -50)},

    # Europe
    {"name": "Western Europe", "lat": (40, 55), "lon": (-10, 10)},
    {"name": "Eastern Europe", "lat": (40, 60), "lon": (10, 40)},
    {"name": "Scandinavia", "lat": (55, 71), "lon": (5, 30)},
    {"name": "Iberian Peninsula", "lat": (36, 44), "lon": (-10, 3)},
    {"name": "British Isles", "lat": (50, 60), "lon": (-10, 2)},

    # Africa
    {"name": "North Africa", "lat": (15, 37), "lon": (-20, 40)},
    {"name": "Sahel", "lat": (10, 20), "lon": (-20, 35)},
    {"name": "Central Africa", "lat": (-10, 10), "lon": (10, 35)},
    {"name": "Southern Africa", "lat": (-35, -10), "lon": (15, 35)},

    # Asia
    {"name": "Middle East", "lat": (20, 40), "lon": (30, 60)},
    {"name": "South Asia", "lat": (5, 30), "lon": (60, 95)},
    {"name": "Southeast Asia", "lat": (-10, 25), "lon": (90, 135)},
    {"name": "East Asia", "lat": (20, 50), "lon": (100, 135)},
    {"name": "Central Asia", "lat": (30, 55), "lon": (60, 100)},
    {"name": "Siberia", "lat": (50, 75), "lon": (60, 160)},

    # Oceania
    {"name": "Australia", "lat": (-50, -10), "lon": (110, 155)},
    {"name": "New Zealand", "lat": (-47, -33), "lon": (165, 180)},

    # Polar
    {"name": "Antarctica", "lat": (-90, -60), "lon": (-180, 180)},
    {"name": "Arctic", "lat": (66, 90), "lon": (-180, 180)},

    # Oceanic Subregions
    {"name": "Caribbean Sea", "lat": (10, 25), "lon": (-85, -60)},
    {"name": "Gulf of Mexico", "lat": (18, 30), "lon": (-97, -81)},
    {"name": "North Atlantic Ocean", "lat": (0, 70), "lon": (-80, 0)},
    {"name": "South Atlantic Ocean", "lat": (-60, 0), "lon": (-70, 20)},
    {"name": "North Pacific Ocean", "lat": (0, 66), "lon": (120, -120)},
    {"name": "South Pacific Ocean", "lat": (-60, 0), "lon": (120, -70)},
    {"name": "Indian Ocean", "lat": (-60, 30), "lon": (20, 120)},
    {"name": "Arctic Ocean", "lat": (66, 90), "lon": (-180, 180)},
    {"name": "Southern Ocean", "lat": (-90, -60), "lon": (-180, 180)},
    {"name": "Mediterranean Sea", "lat": (30, 45), "lon": (-5, 40)}
]

# Step 3: Assign region names to grid cells
for region in regions:
    lat_cond = (region_mask.lat >= region["lat"][0]) & (region_mask.lat <= region["lat"][1])
    lon_cond = (region_mask.lon >= region["lon"][0]) & (region_mask.lon <= region["lon"][1])
    region_mask.loc[lat_cond, lon_cond] = region["name"]

# Step 4: Save for reuse (optional)
region_mask.to_netcdf("region_mask_global.nc")

# Step 5: Example use with a weather dataset
# Assuming weather_ds is your xarray.Dataset with lat/lon dimensions
# weather_ds = xr.open_dataset("your_weather_data.nc")
# weather_ds = weather_ds.assign_coords(region=region_mask)

# Step 6: Example grouping by region or subregion
# regional_mean = weather_ds['t2m'].groupby("region").mean(dim=["lat", "lon"])
