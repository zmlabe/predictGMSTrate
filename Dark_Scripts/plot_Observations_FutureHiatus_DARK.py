"""
Plots a time series of monthly temperatures in the Arctic

Author : Zachary Labe
Date : 20 January 2020
"""

import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap, addcyclic, shiftgrid
import numpy as np
import cmocean
import cmasher as cmr
import palettable.cubehelix as cm
from netCDF4 import Dataset
import nclcmaps as ncm
import scipy.stats as sts
import calc_Utilities as UT

### Define constants
directorydata = '/Users/zlabe/Data/ERA5/monthly/'       
directorydataoutput = '/Users/zlabe/Documents/SciComm/ERA5_ClimateMaps/Data/'            
directoryfigure = '/Users/zlabe/Documents/Projects/predictGMSTrate/Dark_Figures/'
allyears = np.arange(1950,2021+1,1)
monq = [r'JAN',r'FEB',r'MAR',r'APR',r'MAY',r'JUN',r'JUL',r'AUG',r'SEP',r'OCT',r'NOV',r'DEC']

### Read in data
years = np.arange(1979,2021+1,1)
data = Dataset(directorydata + 'T2M_1979-2021.nc')
lat = data.variables['latitude'][:]
lon = data.variables['longitude'][:]
tempq = np.nanmean(data.variables['T2M'][:],axis=1) # for ERA5T
data.close()

### Selecting 2020 and 2021 data to add to 1979-2019 file
empty = np.empty((years.shape[0]*12-len(tempq),lat.shape[0],lon.shape[0]))
empty[:] = np.nan
yr2021 = np.append(tempq[-12+len(empty):,:,:],empty,axis=0)
temp = np.reshape(tempq[:-12+len(empty)],(tempq.shape[0]//12,12,lat.shape[0],lon.shape[0]))
tempqq = np.append(temp,yr2021[np.newaxis,:,:,:],axis=0)
recent20202021 = tempqq[-2:,:,:,:]

### Read in data
years50 = np.arange(1950,2019+1,1)
data = Dataset(directorydata + 'T2M_1950-2019.nc')
lats = data.variables['latitude'][:]
lons = data.variables['longitude'][:]
tempold = data.variables['T2M'][:]
data.close()
tempshape = np.reshape(tempold,(years50.shape[0],12,lats.shape[0],lons.shape[0]))

### Combine all data
alldata = np.append(tempshape,recent20202021,axis=0) - 273.15
alldata = np.asarray(alldata)

### Calculate anomalies
base = np.where((allyears >= 1951) & (allyears <= 1980))[0]
climo = np.nanmean(alldata[base,:,:,:],axis=0)
anom = alldata - climo

### Calculate annual mean
anomyr = np.nanmean(anom[:,:,:,:],axis=1)

### Calculate GMST
lon2,lat2 = np.meshgrid(lon,lat)
ave = UT.calc_weightedAve(anomyr,lat2)

### Select 1990
AGWyr = 1990
yr90 = np.where((allyears >= 1990))[0]
ave90 = ave[yr90][:-1]
years90 = allyears[yr90]

### Final points
finaltwo = ave[-2:]

###############################################################################
###############################################################################
###############################################################################
### Read in data for observations
directorydata = '/Users/zlabe/Documents/Research/GmstTrendPrediction/Data/'
savename = 'ANNv2_OHC100_hiatus_relu_L2_0.5_LR_0.001_Batch128_Iters500_2x30_SegSeed24120_NetSeed87750_EnsembleMeanRemoved'
predict_obs = np.genfromtxt(directorydata + 'obsLabels_' + savename + '.txt')
actual_obs = np.genfromtxt(directorydata + 'obsActualLabels_' + savename + '.txt')
confidence = np.genfromtxt(directorydata + 'obsConfid_' + savename + '.txt')

where_hiatusq = np.where(predict_obs == 1)[0]
where_hiatus = years90[where_hiatusq]
 
trendlength = 10 
hiatusSLOPE = 0.01
typeOfTrend = 'hiatus'
data = ave90
yearsnew = np.arange(1990,2020+1,1)
### Calculate trend periods
yearstrend = np.empty((len(yearsnew)-trendlength+1,trendlength))
datatrend = np.empty((len(yearsnew)-trendlength+1,trendlength))
for hi in range(len(yearsnew)-(trendlength-1)):
    yearstrend[hi,:] = np.arange(yearsnew[hi],yearsnew[hi]+trendlength,1)
    datatrend[hi,:] = data[hi:hi+trendlength]

### Calculate trend lines    
linetrend = np.empty((len(yearsnew)-trendlength+1,2))
for hi in range(len(yearsnew)-trendlength+1):         
    linetrend[hi,:] = np.polyfit(yearstrend[hi],datatrend[hi],1)
    
### Count number of hiatus or acceleration periods
slope = linetrend[:,0]     
indexslopeNegative = np.where((slope[:] <= hiatusSLOPE))[0]
print('INDEX OF **%s**---->' % typeOfTrend,indexslopeNegative)

###############################################################################
###############################################################################
###############################################################################               
### Plot Figure
plt.rc('text',usetex=True)
plt.rc('font',**{'family':'sans-serif','sans-serif':['Avant Garde']}) 
plt.rc('savefig',facecolor='black')
plt.rc('axes',edgecolor='darkgrey')
plt.rc('xtick',color='darkgrey')
plt.rc('ytick',color='darkgrey')
plt.rc('axes',labelcolor='darkgrey')
plt.rc('axes',facecolor='black')

### Adjust axes in time series plots 
def adjust_spines(ax, spines):
    for loc, spine in ax.spines.items():
        if loc in spines:
            spine.set_position(('outward', 5))
        else:
            spine.set_color('none')  
    if 'left' in spines:
        ax.yaxis.set_ticks_position('left')
    else:
        ax.yaxis.set_ticks([])

    if 'bottom' in spines:
        ax.xaxis.set_ticks_position('bottom')
    else:
        ax.xaxis.set_ticks([]) 
        
fig = plt.figure(figsize=(9,6))
ax = plt.subplot(211)

adjust_spines(ax, ['left', 'bottom'])            
ax.spines['top'].set_color('none')
ax.spines['right'].set_color('none')
ax.spines['bottom'].set_linewidth(2)
ax.spines['left'].set_linewidth(2)
ax.tick_params('both',length=5.5,width=2,which='major',labelsize=6)

plt.fill_between(x=years90[-11:-1],y1=0,y2=1.1,facecolor='darkgrey',zorder=0,
             alpha=0.3,edgecolor='none')

plt.plot(years90[:-1],ave90,linewidth=5,color='crimson',zorder=1,clip_on=False)
plt.plot(years90[-2:],finaltwo,linewidth=5,color='crimson',zorder=2,
         clip_on=False,linestyle='--',dashes=(0.5,0.5))
plt.scatter(years90[-1],finaltwo[-1],marker='o',s=50,zorder=3,
            color='crimson',clip_on=False)
for hi in range(len(linetrend)):
    if linetrend[hi,0] < 0.01:
        cc = 'deepskyblue'
        ll = 1.5
    else:
        cc = 'w'
        ll = 0.4
    plt.plot(yearstrend[hi,:],linetrend[hi,0]*yearstrend[hi,:]+linetrend[hi,1],
         color=cc,linewidth=ll)

plt.xticks(np.arange(1950,2040,5),np.arange(1950,2040,5))
plt.yticks(np.arange(-5,5.1,0.1),map(str,np.round(np.arange(-5,5.1,0.1),2))) 
plt.xlim([1990,2029])
plt.ylim([0.2,1.1])

plt.ylabel(r'\textbf{GMST Anomaly ($\bf{^\circ}$C)}',fontsize=8,
                      color='w')

plt.tight_layout()
plt.subplots_adjust(hspace=0.4)

plt.savefig(directoryfigure + 'GMST_obs-future_hiatus.png',dpi=600)