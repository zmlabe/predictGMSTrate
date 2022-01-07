"""
Script for calculating the IPO index in each ensemble member

Author     : Zachary M. Labe
Date       : 17 September 2021
Version    : 12
"""

### Import packages
import sys
import math
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
from mpl_toolkits.basemap import Basemap, addcyclic, shiftgrid
import palettable.cubehelix as cm
import palettable.cartocolors.qualitative as cc
import palettable.scientific.sequential as sss
import cmocean as cmocean
import calc_Utilities as UT
import calc_dataFunctions as df
import calc_Stats as dSS
import scipy.stats as sts
import scipy.sparse as sparse
import scipy.sparse.linalg as linalg
import matplotlib
import cmasher as cmr
from eofs.standard import Eof

### Plotting defaults 
plt.rc('text',usetex=True)
plt.rc('font',**{'family':'sans-serif','sans-serif':['Avant Garde']})

###############################################################################
###############################################################################
###############################################################################
### Data preliminaries 
modelGCMs = ['CESM2-LE']
dataset_obs = 'ERA5'
allDataLabels = modelGCMs
letters = ["a","b","c","d","e","f","g","h","i","j","k","l","m"]
datasetsingle = ['CESM2le']
monthlychoiceq = ['annual']
variables = ['SST']
reg_name = 'SMILEGlobe'
level = 'surface'
###############################################################################
###############################################################################
randomalso = False
timeper = 'hiatus'
shuffletype = 'GAUSS'
###############################################################################
###############################################################################
land_only = False
ocean_only = False
###############################################################################
###############################################################################
baseline = np.arange(1951,1980+1,1)
###############################################################################
###############################################################################
window = 0
if window == 0:
    rm_standard_dev = False
    ravel_modelens = False
    ravelmodeltime = False
else:
    rm_standard_dev = True
    ravelmodeltime = False
    ravel_modelens = True
yearsall = np.arange(1979+window,2099+1,1)
yearsobs = np.arange(1979+window,2020+1,1)
###############################################################################
###############################################################################
numOfEns = 40
lentime = len(yearsall)
###############################################################################
###############################################################################
dataset = datasetsingle[0]
lat_bounds,lon_bounds = UT.regions(reg_name)
###############################################################################
###############################################################################
ravelyearsbinary = False
ravelbinary = False
lensalso = True
###############################################################################
###############################################################################
### Processing data steps
rm_ensemble_mean = True
###############################################################################
###############################################################################
###############################################################################
###############################################################################
### Read in model and observational/reanalysis data
def read_primary_dataset(variq,dataset,monthlychoice,numOfEns,lensalso,randomalso,ravelyearsbinary,ravelbinary,shuffletype,timeper,lat_bounds=lat_bounds,lon_bounds=lon_bounds):
    data,lats,lons = df.readFiles(variq,dataset,monthlychoice,numOfEns,lensalso,randomalso,ravelyearsbinary,ravelbinary,shuffletype,timeper)
    datar,lats,lons = df.getRegion(data,lats,lons,lat_bounds,lon_bounds)
    print('\nOur dataset: ',dataset,' is shaped',data.shape)
    return datar,lats,lons
  
def read_obs_dataset(variq,dataset_obs,numOfEns,lensalso,randomalso,ravelyearsbinary,ravelbinary,shuffletype,lat_bounds=lat_bounds,lon_bounds=lon_bounds):
    data_obs,lats_obs,lons_obs = df.readFiles(variq,dataset_obs,monthlychoice,numOfEns,lensalso,randomalso,ravelyearsbinary,ravelbinary,shuffletype,timeper)
    data_obs,lats_obs,lons_obs = df.getRegion(data_obs,lats_obs,lons_obs,lat_bounds,lon_bounds)
    
    print('our OBS dataset: ',dataset_obs,' is shaped',data_obs.shape)
    return data_obs,lats_obs,lons_obs

### Call functions
vv = 0
mo = 0
variq = variables[vv]
monthlychoice = monthlychoiceq[mo]
directoryfigure = '/Users/zlabe/Desktop/GmstTrendPrediction/ANN_v2/PDO/'
saveData =  monthlychoice + '_' + variq + '_' + reg_name + '_' + dataset_obs
print('*Filename == < %s >' % saveData) 

### Read data
models,lats,lons = read_primary_dataset(variq,dataset,monthlychoice,numOfEns,
                                        lensalso,randomalso,ravelyearsbinary,
                                        ravelbinary,shuffletype,timeper,
                                        lat_bounds,lon_bounds)
obs,lats_obs,lons_obs = read_obs_dataset(variq,dataset_obs,numOfEns,lensalso,randomalso,ravelyearsbinary,ravelbinary,shuffletype,lat_bounds=lat_bounds,lon_bounds=lon_bounds)

### Slice for number of years to begin hiatus
AGWstart = 1990
yearsq_m = np.where((yearsall >= AGWstart))[0]
yearsq_o = np.where((yearsobs >= AGWstart))[0]
models_slice = models[:,yearsq_m,:,:]
obs_slice = obs[yearsq_o,:,:]

### Calculate global mean temperature
lon2,lat2 = np.meshgrid(lons,lats)
modelsm = UT.calc_weightedAve(models_slice,lat2)
obsm = UT.calc_weightedAve(obs_slice,lat2)

###############################################################################          
### Calculate ensemble spread statistics
meaens = np.nanmean(modelsm[:,:],axis=0)
maxens = np.nanmax(modelsm[:,:],axis=0)
minens = np.nanmin(modelsm[:,:],axis=0)
spread = maxens - minens

###############################################################################  
### Remove ensemble mean
if rm_ensemble_mean == True:
    models_var = dSS.remove_ensemble_mean(models_slice,ravel_modelens,
                                          ravelmodeltime,rm_standard_dev,
                                          numOfEns)

###############################################################################  
###############################################################################  
###############################################################################
### Calculate SST regions
region1 = models_var.copy() #  25°N–45°N, 140°E–145°W
latr1 = np.where((lats >= 25) & (lats <= 45))[0]
lonr1 = np.where((lons >= 140) & (lons <= (180+(180-145))))[0]
latRegion1 = lats[latr1]
lonRegion1 = lons[lonr1]
lon2r1,lat2r1 = np.meshgrid(lonRegion1,latRegion1)
sstregion1lat = region1[:,:,latr1,:]
sstregion1 = sstregion1lat[:,:,:,lonr1]
mean_r1 = UT.calc_weightedAve(sstregion1,lat2r1)

region2 = models_var.copy() #  10°S–10°N, 170°E–90°W
latr2 = np.where((lats >= -10) & (lats <= 10))[0]
lonr2 = np.where((lons >= 170) & (lons <= (180+(180-90))))[0]
latRegion2 = lats[latr2]
lonRegion2 = lons[lonr2]
lon2r2,lat2r2 = np.meshgrid(lonRegion2,latRegion2)
sstregion2lat = region2[:,:,latr2,:]
sstregion2 = sstregion2lat[:,:,:,lonr2]
mean_r2 = UT.calc_weightedAve(sstregion2,lat2r2)

region3 = models_var.copy() #  50°S–15°S, 150°E–160°W
latr3 = np.where((lats >= -50) & (lats <= -15))[0]
lonr3 = np.where((lons >= 150) & (lons <= (180+(180-160))))[0]
latRegion3 = lats[latr3]
lonRegion3 = lons[lonr3]
lon2r3,lat2r3 = np.meshgrid(lonRegion3,latRegion3)
sstregion3lat = region3[:,:,latr3,:]
sstregion3 = sstregion3lat[:,:,:,lonr3]
mean_r3 = UT.calc_weightedAve(sstregion3,lat2r3)

###############################################################################
### Calculate IPO
IPOindex = mean_r2 - ((mean_r1 + mean_r3)/2)
IPOindexz = (IPOindex - np.mean(IPOindex))/np.std(IPOindex)

### Save IPO index
directoryoutput = '/Users/zlabe/Documents/Research/GmstTrendPrediction/Data/IPO/'
np.savetxt(directoryoutput + 'IPO_CESM2LE_1990-2099.txt',IPOindexz)

###############################################################################
###############################################################################
###############################################################################
###############################################################################
### Read in PDO index
directorydata = '/Users/zlabe/Documents/Research/GmstTrendPrediction/Data/'
PDO = np.genfromtxt(directorydata + '/PDO/PDO_CESM2LE_1990-2099.txt',unpack=True).transpose()

###############################################################################
### Compare PDO and IPO correlations
corr = np.empty((models_var.shape[0]))
p = np.empty((models_var.shape[0]))
for i in range(models_var.shape[0]):
    corr[i],p[i] = sts.pearsonr(IPOindexz[i],PDO[i])
    
###############################################################################
###############################################################################
###############################################################################
### Create plot of correlations for the ensembles
fig = plt.figure()
ax = plt.subplot(111)

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

adjust_spines(ax, ['left', 'bottom'])
ax.spines['top'].set_color('none')
ax.spines['right'].set_color('none')
ax.spines['left'].set_color('dimgrey')
ax.spines['bottom'].set_color('dimgrey')
ax.spines['left'].set_linewidth(2)
ax.spines['bottom'].set_linewidth(2)
ax.tick_params('both',length=4,width=2,which='major',color='dimgrey')
ax.tick_params(axis='x',labelsize=7,pad=4)
ax.tick_params(axis='y',labelsize=7,pad=4)
ax.yaxis.grid(zorder=1,color='dimgrey',alpha=0.35,clip_on=False)

plt.plot(corr,color='maroon',linewidth=3,clip_on=False,alpha=1,
         marker='o',markersize=7,zorder=10)

plt.xlabel(r'\textbf{CESM2-LE Ensemble Member \#}',fontsize=8,color='dimgrey')
plt.ylabel(r'\textbf{IPO \& PDO: Correlation Coefficient}',fontsize=8,color='dimgrey')
plt.yticks(np.arange(0,1.1,0.1),map(str,np.round(np.arange(0,1.1,0.1),2)))
plt.xticks(np.arange(0,39+1,3),map(str,np.arange(1,40+1,3)))
plt.xlim([0,39])   
plt.ylim([0,1])

plt.subplots_adjust(bottom=0.15)
plt.savefig(directoryfigure + 'IPO-PDO_CorrelationCESM2le.png',
            dpi=600)
                 