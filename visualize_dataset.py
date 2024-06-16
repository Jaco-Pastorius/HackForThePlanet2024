import netCDF4 as nc
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature


# Data set path 
path2data = r"C:\Users\gabma\Dropbox\PC\Documents\Master\HackForThePlanet\oco3_China.nc4"

# Open the .nc4 file using xarray
dataset = xr.open_dataset(path2data)

latitude = dataset['latitude']
longitude = dataset['longitude']
xco2 = dataset['xco2']

plt.figure(figsize=(12, 6))
ax = plt.axes(projection=ccrs.PlateCarree())

# Plot the data
sc = plt.scatter(longitude, latitude, c=xco2, cmap='coolwarm', s=1, transform=ccrs.PlateCarree(), alpha=0.5)
plt.colorbar(sc, label='XCO2 (ppm)')

# Add coastlines and borders for context
ax.coastlines()
ax.add_feature(cfeature.BORDERS, linestyle=':')

# Add a title
plt.title('CO2 Concentration Over China')

# Show the plot
plt.show()