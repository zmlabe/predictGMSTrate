"""
Script calculates example of OHC map

Author     : Zachary M. Labe
Date       : 5 October 2021
Version    : 1 
"""

### Import packages
import sys
import math
import time
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
from mpl_toolkits.basemap import Basemap, addcyclic, shiftgrid
import palettable.cubehelix as cm
import palettable.cartocolors.qualitative as cc
import palettable.scientific.sequential as sss
import cmocean as cmocean
import calc_Utilities as UT
import matplotlib
import cmasher as cmr
from netCDF4 import Dataset

### Plotting defaults 
plt.rc('text',usetex=True)
plt.rc('font',**{'family':'sans-serif','sans-serif':['Avant Garde']})

### Preliminaries
directorydata = '/Users/zlabe/Data/CESM2-LE/monthly/OHC100/'
directoryfigure = '/Users/zlabe/Desktop/GmstTrendPrediction/ANN_v2/Raw/'

###############################################################################
###############################################################################
###############################################################################
### Read in data
ens = 10
months = 12*100
ohcq = np.empty((ens,months,96,144))
for i in range(1,ens+1,1):
    filename = 'OHC100_%s_1850-2100.nc' % i 
    data = Dataset(directorydata + filename)
    time = data.variables['time'][:]
    lats = data.variables['latitude'][:]
    lons = data.variables['longitude'][:]
    ohcq[i-1,:,:,:] = data.variables['OHC100'][:months,:,:]
    data.close()
    
### Calculate annual mean
ohcnew = ohcq.reshape(ens,months//12,12,lats.shape[0],lons.shape[0])
ohcannual = np.nanmean(ohcnew,axis=2)

### Remove ensemble mean
ensmean = np.nanmean(ohcannual,axis=0)
rmohc = ohcannual - ensmean

### Standardize
mean = np.nanmean(rmohc,axis=1)
std = np.nanstd(rmohc,axis=1)
z = (rmohc - mean[:,np.newaxis,:,:])/std[:,np.newaxis,:,:]

### Pick example
ohc = z[0,50,:,:]

###############################################################################
###############################################################################
### Plot of sample ohc 
limit = np.arange(-2,2.01,0.01)
cmap = cmocean.cm.balance

fig = plt.figure(figsize=(10,5))
###############################################################################
ax1 = plt.subplot(111)
m = Basemap(projection='robin',lon_0=-180,resolution='l',area_thresh=10000)
m.drawcoastlines(color='darkgrey',linewidth=0.75)
    
### Variable
varn = ohc
lons = np.where(lons >180,lons-360,lons)
x, y = np.meshgrid(lons,lats)
   
circle = m.drawmapboundary(fill_color='dimgrey',color='dimgray',
                  linewidth=0.7)
circle.set_clip_on(False)

cs1 = m.contourf(x,y,varn,limit,extend='both',latlon=True)
cs1.set_cmap(cmap) 
m.fillcontinents(color='dimgrey',lake_color='dimgrey')

plt.tight_layout()
plt.savefig(directoryfigure + 'SampleMap_OHC.png',dpi=1000)

###############################################################################
###############################################################################
### Plot of sample LRP
data = Dataset('/Users/zlabe/Documents/Research/GmstTrendPrediction/Data/LRPMapTesting_ANNv2_OHC100_hiatus_relu_L2_0.5_LR_0.001_Batch128_Iters500_2x30_SegSeed24120_NetSeed87750_EnsembleMeanRemoved.nc')
lrp = data.variables['LRP'][:]
lats = data.variables['lat'][:]
lons = data.variables['lon'][:]
data.close()

limit = np.arange(0,0.71,0.01)
cmap = cm.cubehelix2_16.mpl_colormap

fig = plt.figure(figsize=(10,5))
###############################################################################
ax1 = plt.subplot(111)
m = Basemap(projection='robin',lon_0=-180,resolution='l',area_thresh=10000)
m.drawcoastlines(color='darkgrey',linewidth=0.75)
    
### Variable
varn = lrp[0]
lons = np.where(lons >180,lons-360,lons)
x, y = np.meshgrid(lons,lats)
   
circle = m.drawmapboundary(fill_color='dimgrey',color='dimgray',
                  linewidth=0.7)
circle.set_clip_on(False)

cs1 = m.contourf(x,y,varn,limit,extend='both',latlon=True)
cs1.set_cmap(cmap) 
m.fillcontinents(color='dimgrey',lake_color='dimgrey')

plt.tight_layout()
plt.savefig(directoryfigure + 'SampleMap_LRP.png',dpi=1000)
