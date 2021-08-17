"""
Script for plotting global mean surface temperature from 1850 to 2100

Author     : Zachary M. Labe
Date       : 3 August 2021
Version    : 1 
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
import matplotlib
import cmasher as cmr

### Plotting defaults 
plt.rc('text',usetex=True)
plt.rc('font',**{'family':'sans-serif','sans-serif':['Avant Garde']}) 
plt.rc('savefig',facecolor='black')
plt.rc('axes',edgecolor='darkgrey')
plt.rc('xtick',color='darkgrey')
plt.rc('ytick',color='darkgrey')
plt.rc('axes',labelcolor='darkgrey')
plt.rc('axes',facecolor='black')

###############################################################################
###############################################################################
###############################################################################
### Data preliminaries 
modelGCMs = ['CESM2-LE']
dataset_obs = '20CRv3'
allDataLabels = modelGCMs
letters = ["a","b","c","d","e","f","g","h","i","j","k","l","m"]
datasetsingle = ['CESM2le']
monthlychoiceq = ['annual']
variables = ['T2M']
reg_name = 'SMILEGlobe'
level = 'surface'
###############################################################################
###############################################################################
randomalso = False
timeper = 'all'
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
yearsall = np.arange(1850+window,2100+1,1)
yearsobs = np.arange(1850+window,2015+1,1)
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
directoryfigure = '/Users/zlabe/Documents/Projects/predictGMSTrate/Dark_Figures/'
# saveData =  monthlychoice + '_' + variq + '_' + reg_name + '_' + dataset_obs
# print('*Filename == < %s >' % saveData) 

# ### Read data
# models,lats,lons = read_primary_dataset(variq,dataset,monthlychoice,numOfEns,
#                                         lensalso,randomalso,ravelyearsbinary,
#                                         ravelbinary,shuffletype,timeper,
#                                         lat_bounds,lon_bounds)
# obs,lats_obs,lons_obs = read_obs_dataset(variq,dataset_obs,numOfEns,lensalso,randomalso,ravelyearsbinary,ravelbinary,shuffletype,lat_bounds=lat_bounds,lon_bounds=lon_bounds)

# ### Calculate global mean temperature
# lon2,lat2 = np.meshgrid(lons,lats)
# modelsm = UT.calc_weightedAve(models,lat2)
# obsm = UT.calc_weightedAve(obs,lat2)

# ###############################################################################          
# ### Calculate ensemble spread statistics
# meaens = np.nanmean(modelsm[:,:],axis=0)
# maxens = np.nanmax(modelsm[:,:],axis=0)
# minens = np.nanmin(modelsm[:,:],axis=0)
# spread = maxens - minens

###############################################################################
###############################################################################
###############################################################################
### Create time series
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
ax.spines['left'].set_color('darkgrey')
ax.spines['bottom'].set_color('darkgrey')
ax.spines['left'].set_linewidth(2)
ax.spines['bottom'].set_linewidth(2)
ax.tick_params('both',length=4,width=2,which='major',color='darkgrey')
ax.tick_params(axis='x',labelsize=7,pad=4)
ax.tick_params(axis='y',labelsize=7,pad=4)

plt.plot(yearsall,meaens[:],'-',
            color='deepskyblue',linewidth=2,clip_on=False,alpha=1,zorder=3)
plt.plot(yearsobs,obsm,'--',dashes=(1,0.3),
            color='gold',linewidth=1,clip_on=False,alpha=1,zorder=5)
plt.plot(np.tile(yearsall,(40,1)).transpose(),modelsm.transpose(),color='w',linewidth=0.2,alpha=0.2)
plt.plot(yearsall,modelsm[7],color='r',linewidth=0.5,alpha=1,zorder=4)
# ax.fill_between(yearsall,minens[:],maxens[:],facecolor='w',alpha=0.25,zorder=1,clip_on=False)

plt.text(yearsall[-1]+2,meaens[-1],r'\textbf{%s}' % modelGCMs[0],
              color='deepskyblue',fontsize=8,ha='left',va='center')

plt.ylabel(r'\textbf{GMST [$^{\circ}$C]}',fontsize=10,color='darkgrey')
plt.yticks(np.arange(0,21,0.5),map(str,np.round(np.arange(0,21,0.5),2)))
plt.xticks(np.arange(1850,2100+1,25),map(str,np.arange(1850,2100+1,25)))
plt.xlim([1850,2100])   
plt.ylim([13.5,19])

plt.subplots_adjust(bottom=0.15)
plt.savefig(directoryfigure + 'TimeSeries_MeanGlobalTemperature_CESM2LEonly_DARK.png',
            dpi=600)