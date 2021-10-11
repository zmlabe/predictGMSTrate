"""
ANNv2 for making raw OHC maps of observations

Author     : Zachary M. Labe
Date       : 11 October 2021
Version    : 2(mostly for testing)
"""

### Import packages
import sys
import matplotlib.pyplot as plt
import numpy as np
from netCDF4 import Dataset
import scipy.stats as stats
from mpl_toolkits.basemap import Basemap, addcyclic, shiftgrid
import palettable.cubehelix as cm
import cmocean as cmocean
import calc_Utilities as UT
import calc_dataFunctions as df
import calc_Stats as dSS
from sklearn.metrics import accuracy_score

### Plotting defaults 
plt.rc('text',usetex=True)
plt.rc('font',**{'family':'sans-serif','sans-serif':['Avant Garde']}) 
plt.rc('savefig',facecolor='black')
plt.rc('axes',edgecolor='darkgrey')
plt.rc('xtick',color='darkgrey')
plt.rc('ytick',color='darkgrey')
plt.rc('axes',labelcolor='darkgrey')
plt.rc('axes',facecolor='black')

### Paths
directoryfigure = '/Users/zlabe/Documents/Projects/predictGMSTrate/Dark_Figures/'

### Hyperparamters for files of the ANN model
rm_ensemble_mean = True

### Read in data for OHC composites
directorydata = '/Users/zlabe/Documents/Research/GmstTrendPrediction/Data/OHC_comp/'
data = Dataset(directorydata + 'OHCcomp_OBS_hiatus_True.nc')
lat = data.variables['lat'][:]
lon = data.variables['lon'][:]
ohc_real = data.variables['OHC100'][:]
data.close()

directorydata = '/Users/zlabe/Documents/Research/GmstTrendPrediction/Data/OHC_comp/'
data = Dataset(directorydata + 'OHCcomp_OBS_hiatus_Future.nc')
ohc_future = data.variables['OHC100'][:]
data.close()

### Read in LRP
directorydatal = '/Users/zlabe/Documents/Research/GmstTrendPrediction/Data/'
savename = 'obs_ANNv2_OHC100_hiatus_relu_L2_0.5_LR_0.001_Batch128_Iters500_2x30_SegSeed24120_NetSeed87750_EnsembleMeanRemoved'
datalrp = Dataset(directorydatal + 'LRP_comp/LRPMap_comp_' + 'correcthiatus' + '_' + savename + '.nc')
lrp_real = datalrp.variables['LRP'][:]
datalrp.close()

datalrp = Dataset(directorydatal + 'LRP_comp/LRPMap_comp_' + 'wronghiatus' + '_' + savename + '.nc')
lrp_future = datalrp.variables['LRP'][:]
datalrp.close()

###############################################################################
###############################################################################
###############################################################################
### Plot subplot of observations
letters = ["a","b","c","d","e","f","g","h","i","j","k","l","m","n"]
if rm_ensemble_mean == False:
    limit = np.arange(-1.5,1.51,0.02)
    barlim = np.round(np.arange(-1.5,1.6,0.5),2)
elif rm_ensemble_mean == True:
    limit = np.arange(-1.5,1.6,0.02)
    barlim = np.round(np.arange(-1.5,1.6,0.5),2)
cmap = cmocean.cm.balance
label = r'\textbf{OHC100 [ Normalized ]}'

fig = plt.figure(figsize=(8,3))
###############################################################################
ax1 = plt.subplot(121)
m = Basemap(projection='robin',lon_0=-180,resolution='l',area_thresh=10000)
m.drawcoastlines(color='w',linewidth=0.15)
    
### Variable
lon = np.where(lon >180,lon-360,lon)
x, y = np.meshgrid(lon,lat)
   
circle = m.drawmapboundary(fill_color='darkgrey',color='dimgray',
                  linewidth=0.7)
circle.set_clip_on(False)

cs1 = m.contourf(x,y,ohc_real,limit,extend='both',latlon=True)
cs1.set_cmap(cmap) 

csc = m.contour(x,y,lrp_real,np.arange(0.15,0.16,0.1),linestyles='-',latlon=True,
                colors='gold',linewidths=1)

# ### Box 1
# la1 = 25
# la2 = 45
# lo1 = 140
# lo2 = 180+(180-145)
# lonsslice = np.linspace(lo1,lo2,lo2-lo1+1)
# latsslice = np.ones(len(lonsslice))*la2
# m.plot(lonsslice, latsslice, color='aqua', linewidth=1.5, latlon=True,zorder=4)
# latsslice = np.ones(len(lonsslice))*la1
# m.plot(lonsslice, latsslice, color='aqua', linewidth=1.5, latlon=True,zorder=4)
# m.drawgreatcircle(lo1, la1, lo1, la2,linewidth=1.5,color='aqua',zorder=4)
# m.drawgreatcircle(lo2, la2, lo2, la1,linewidth=1.5,color='aqua',zorder=4)

# ### Box 2
# la1 = -10
# la2 = 10
# lo1 = 170
# lo2 = 180+(180-90)
# lonsslice = np.linspace(lo1,lo2,lo2-lo1+1)
# latsslice = np.ones(len(lonsslice))*la2
# m.plot(lonsslice, latsslice, color='aqua', linewidth=1.5, latlon=True,zorder=4)
# latsslice = np.ones(len(lonsslice))*la1
# m.plot(lonsslice, latsslice, color='aqua', linewidth=1.5, latlon=True,zorder=4)
# m.drawgreatcircle(lo1, la1, lo1, la2,linewidth=1.5,color='aqua',zorder=4)
# m.drawgreatcircle(lo2, la2, lo2, la1,linewidth=1.5,color='aqua',zorder=4)

# ### Box 3
# la1 = -50
# la2 = -15
# lo1 = 150
# lo2 = 180+(180-160)
# lonsslice = np.linspace(lo1,lo2,lo2-lo1+1)
# latsslice = np.ones(len(lonsslice))*la2
# m.plot(lonsslice, latsslice, color='aqua', linewidth=1.5, latlon=True,zorder=4)
# latsslice = np.ones(len(lonsslice))*la1
# m.plot(lonsslice, latsslice, color='aqua', linewidth=1.5, latlon=True,zorder=4)
# m.drawgreatcircle(lo1, la1, lo1, la2,linewidth=1.5,color='aqua',zorder=4)
# m.drawgreatcircle(lo2, la2, lo2, la1,linewidth=1.5,color='aqua',zorder=4)

m.fillcontinents(color='k',lake_color='k')

plt.title(r'\textbf{ACTUAL SLOWDOWN}',fontsize=17,color='darkgrey')       
ax1.annotate(r'\textbf{[%s]}' % letters[0],xy=(0,0),xytext=(0.98,0.84),
              textcoords='axes fraction',color='k',fontsize=9,
              rotation=0,ha='center',va='center')

###############################################################################
ax2 = plt.subplot(122)
m = Basemap(projection='robin',lon_0=-180,resolution='l',area_thresh=10000)
m.drawcoastlines(color='w',linewidth=0.15)

### Variable   
circle = m.drawmapboundary(fill_color='k',color='k',
                  linewidth=0.7)
circle.set_clip_on(False)

cs2 = m.contourf(x,y,ohc_future,limit,extend='both',latlon=True)
cs2.set_cmap(cmap) 

csc = m.contour(x,y,lrp_future,np.arange(0.15,0.16,0.1),linestyles='-',latlon=True,
                colors='gold',linewidths=1)

# ### Box 1
# la1 = 25
# la2 = 45
# lo1 = 140
# lo2 = 180+(180-145)
# lonsslice = np.linspace(lo1,lo2,lo2-lo1+1)
# latsslice = np.ones(len(lonsslice))*la2
# m.plot(lonsslice, latsslice, color='aqua', linewidth=1.5, latlon=True,zorder=4)
# latsslice = np.ones(len(lonsslice))*la1
# m.plot(lonsslice, latsslice, color='aqua', linewidth=1.5, latlon=True,zorder=4)
# m.drawgreatcircle(lo1, la1, lo1, la2,linewidth=1.5,color='aqua',zorder=4)
# m.drawgreatcircle(lo2, la2, lo2, la1,linewidth=1.5,color='aqua',zorder=4)

# ### Box 2
# la1 = -10
# la2 = 10
# lo1 = 170
# lo2 = 180+(180-90)
# lonsslice = np.linspace(lo1,lo2,lo2-lo1+1)
# latsslice = np.ones(len(lonsslice))*la2
# m.plot(lonsslice, latsslice, color='aqua', linewidth=1.5, latlon=True,zorder=4)
# latsslice = np.ones(len(lonsslice))*la1
# m.plot(lonsslice, latsslice, color='aqua', linewidth=1.5, latlon=True,zorder=4)
# m.drawgreatcircle(lo1, la1, lo1, la2,linewidth=1.5,color='aqua',zorder=4)
# m.drawgreatcircle(lo2, la2, lo2, la1,linewidth=1.5,color='aqua',zorder=4)

# ### Box 3
# la1 = -50
# la2 = -15
# lo1 = 150
# lo2 = 180+(180-160)
# lonsslice = np.linspace(lo1,lo2,lo2-lo1+1)
# latsslice = np.ones(len(lonsslice))*la2
# m.plot(lonsslice, latsslice, color='aqua', linewidth=1.5, latlon=True,zorder=4)
# latsslice = np.ones(len(lonsslice))*la1
# m.plot(lonsslice, latsslice, color='aqua', linewidth=1.5, latlon=True,zorder=4)
# m.drawgreatcircle(lo1, la1, lo1, la2,linewidth=1.5,color='aqua',zorder=4)
# m.drawgreatcircle(lo2, la2, lo2, la1,linewidth=1.5,color='aqua',zorder=4)
# m.fillcontinents(color='k',lake_color='k')


plt.title(r'\textbf{FUTURE SLOWDOWN}',fontsize=17,color='darkgrey')       
ax2.annotate(r'\textbf{[%s]}' % letters[1],xy=(0,0),xytext=(0.98,0.84),
              textcoords='axes fraction',color='k',fontsize=9,
              rotation=0,ha='center',va='center')

###############################################################################
cbar_ax1 = fig.add_axes([0.40,0.1,0.2,0.05])                
cbar1 = fig.colorbar(cs1,cax=cbar_ax1,orientation='horizontal',
                    extend='max',extendfrac=0.07,drawedges=False)
cbar1.set_label(label,fontsize=6,color='darkgrey',labelpad=1.4)  
cbar1.set_ticks(barlim)
cbar1.set_ticklabels(list(map(str,barlim)))
cbar1.ax.tick_params(axis='x', size=.01,labelsize=4)
cbar1.outline.set_edgecolor('darkgrey')

plt.tight_layout()
plt.subplots_adjust(hspace=0.4)
plt.savefig(directoryfigure + 'OHC100_Observations_DARK.png',dpi=600)