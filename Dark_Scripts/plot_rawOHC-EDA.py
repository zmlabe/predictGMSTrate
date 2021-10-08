"""
Script looks at raw composites of ocean heat content

Author     : Zachary M. Labe
Date       : 2 September 2021
Version    : 1 
"""

### Import packages
import sys
import math
import time
import matplotlib.pyplot as plt
import numpy as np
import calc_Hiatus_v3 as HA
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

###############################################################################
###############################################################################
###############################################################################
### Data preliminaries 
modelGCMs = ['CESM2le']
dataset_obs = 'OHC'
allDataLabels = modelGCMs
letters = ["a","b","c","d","e","f","g","h","i","j","k","l","m"]
datasetsingle = ['CESM2le']
monthlychoiceq = ['annual']
variables = ['OHC100']
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
directoryfigure = '/Users/zlabe/Desktop/GmstTrendPrediction/RawDataComposites/'
saveData =  monthlychoice + '_' + variq + '_' + reg_name + '_' + dataset_obs
print('*Filename == < %s >' % saveData) 

### Read data
models,lats,lons = read_primary_dataset(variq,dataset,monthlychoice,numOfEns,
                                        lensalso,randomalso,ravelyearsbinary,
                                        ravelbinary,shuffletype,timeper,
                                        lat_bounds,lon_bounds)
obs,lats_obs,lons_obs = read_obs_dataset(variq,dataset_obs,numOfEns,lensalso,randomalso,ravelyearsbinary,ravelbinary,shuffletype,lat_bounds=lat_bounds,lon_bounds=lon_bounds)

modelshape = models.reshape(models.shape[0]*models.shape[1],models.shape[2]*models.shape[3])
mean = np.nanmean(modelshape,axis=0)
std = np.nanstd(modelshape,axis=0)
modelz = (modelshape-mean)/std
modelzreg = modelz.reshape(models.shape)

### Pick example ensemble
enspick = models[0,:,:,:]/1e11
enspickstd = modelzreg[0,:,:,:]

###############################################################################
###############################################################################
###############################################################################
### Plot subplot of observations
letters = ["a","b","c","d","e","f","g","h","i","j","k","l","m","n"]
limit = np.arange(1,1.31,0.01)
barlim = np.round(np.arange(1,1.31,0.1),2)
cmap = cmr.guppy_r
label = r'\textbf{OHC100 - [ Joules $\times$10$^{11}$]}'

fig = plt.figure(figsize=(8,3))
###############################################################################
ax1 = plt.subplot(121)
m = Basemap(projection='moll',lon_0=0,resolution='l',area_thresh=10000)
m.drawcoastlines(color='darkgrey',linewidth=0.27)
    
### Variable
historic = enspick[0]

var, lons_cyclic = addcyclic(historic, lons)
var, lons_cyclic = shiftgrid(180., var, lons_cyclic, start=False)
lon2d, lat2d = np.meshgrid(lons_cyclic, lats)
x, y = m(lon2d, lat2d)
   
circle = m.drawmapboundary(fill_color='dimgrey',color='dimgray',
                  linewidth=0.7)
circle.set_clip_on(False)

cs1 = m.contourf(x,y,var,limit,extend='both')
cs1.set_cmap(cmap) 
m.fillcontinents(color='dimgrey',lake_color='dimgrey')

plt.title(r'\textbf{1979}',fontsize=17,color='dimgrey')      
ax1.annotate(r'\textbf{[%s]}' % letters[0],xy=(0,0),xytext=(0.98,0.84),
              textcoords='axes fraction',color='k',fontsize=9,
              rotation=0,ha='center',va='center')

###############################################################################
ax2 = plt.subplot(122)
m = Basemap(projection='moll',lon_0=0,resolution='l',area_thresh=10000)
m.drawcoastlines(color='darkgrey',linewidth=0.27)

### Variable
future = enspick[-1]
    
var, lons_cyclic = addcyclic(future, lons)
var, lons_cyclic = shiftgrid(180., var, lons_cyclic, start=False)
lon2d, lat2d = np.meshgrid(lons_cyclic, lats)
x, y = m(lon2d, lat2d)
   
circle = m.drawmapboundary(fill_color='dimgrey',color='dimgray',
                  linewidth=0.7)
circle.set_clip_on(False)

cs2 = m.contourf(x,y,var,limit,extend='both')
cs2.set_cmap(cmap) 
m.fillcontinents(color='dimgrey',lake_color='dimgrey')


plt.title(r'\textbf{2099}',fontsize=17,color='dimgrey')      
ax2.annotate(r'\textbf{[%s]}' % letters[1],xy=(0,0),xytext=(0.98,0.84),
              textcoords='axes fraction',color='k',fontsize=9,
              rotation=0,ha='center',va='center')

###############################################################################
cbar_ax1 = fig.add_axes([0.40,0.1,0.2,0.05])                
cbar1 = fig.colorbar(cs1,cax=cbar_ax1,orientation='horizontal',
                    extend='both',extendfrac=0.07,drawedges=False)
cbar1.set_label(label,fontsize=6,color='dimgrey',labelpad=1.4)  
cbar1.set_ticks(barlim)
cbar1.set_ticklabels(list(map(str,barlim)))
cbar1.ax.tick_params(axis='x', size=.01,labelsize=4)
cbar1.outline.set_edgecolor('dimgrey')

plt.tight_layout()
plt.savefig(directoryfigure + 'OHC_ensembleComparison.png',dpi=300)

###############################################################################
###############################################################################
###############################################################################
### Plot subplot of observations standardized
letters = ["a","b","c","d","e","f","g","h","i","j","k","l","m","n"]
limit = np.arange(-3,3.01,0.01)
barlim = np.round(np.arange(-3,4,1),2)
cmap = cmocean.cm.balance
label = r'\textbf{OHC100 - [ standardized ]}'

fig = plt.figure(figsize=(8,3))
###############################################################################
ax1 = plt.subplot(121)
m = Basemap(projection='moll',lon_0=0,resolution='l',area_thresh=10000)
m.drawcoastlines(color='darkgrey',linewidth=0.27)
    
### Variable
historic = enspickstd[0]

var, lons_cyclic = addcyclic(historic, lons)
var, lons_cyclic = shiftgrid(180., var, lons_cyclic, start=False)
lon2d, lat2d = np.meshgrid(lons_cyclic, lats)
x, y = m(lon2d, lat2d)
   
circle = m.drawmapboundary(fill_color='dimgrey',color='dimgray',
                  linewidth=0.7)
circle.set_clip_on(False)

cs1 = m.contourf(x,y,var,limit,extend='both')
cs1.set_cmap(cmap) 
m.fillcontinents(color='dimgrey',lake_color='dimgrey')

plt.title(r'\textbf{1979}',fontsize=17,color='dimgrey')      
ax1.annotate(r'\textbf{[%s]}' % letters[0],xy=(0,0),xytext=(0.98,0.84),
              textcoords='axes fraction',color='k',fontsize=9,
              rotation=0,ha='center',va='center')

###############################################################################
ax2 = plt.subplot(122)
m = Basemap(projection='moll',lon_0=0,resolution='l',area_thresh=10000)
m.drawcoastlines(color='darkgrey',linewidth=0.27)

### Variable
future = enspickstd[-1]
    
var, lons_cyclic = addcyclic(future, lons)
var, lons_cyclic = shiftgrid(180., var, lons_cyclic, start=False)
lon2d, lat2d = np.meshgrid(lons_cyclic, lats)
x, y = m(lon2d, lat2d)
   
circle = m.drawmapboundary(fill_color='dimgrey',color='dimgray',
                  linewidth=0.7)
circle.set_clip_on(False)

cs2 = m.contourf(x,y,var,limit,extend='both')
cs2.set_cmap(cmap) 
m.fillcontinents(color='dimgrey',lake_color='dimgrey')


plt.title(r'\textbf{2099}',fontsize=17,color='dimgrey')      
ax2.annotate(r'\textbf{[%s]}' % letters[1],xy=(0,0),xytext=(0.98,0.84),
              textcoords='axes fraction',color='k',fontsize=9,
              rotation=0,ha='center',va='center')

###############################################################################
cbar_ax1 = fig.add_axes([0.40,0.1,0.2,0.05])                
cbar1 = fig.colorbar(cs1,cax=cbar_ax1,orientation='horizontal',
                    extend='both',extendfrac=0.07,drawedges=False)
cbar1.set_label(label,fontsize=6,color='dimgrey',labelpad=1.4)  
cbar1.set_ticks(barlim)
cbar1.set_ticklabels(list(map(str,barlim)))
cbar1.ax.tick_params(axis='x', size=.01,labelsize=4)
cbar1.outline.set_edgecolor('dimgrey')

plt.tight_layout()
plt.savefig(directoryfigure + 'OHC_ensembleComparison_Z.png',dpi=300)

###############################################################################
###############################################################################
###############################################################################        
### Calculate ensemble spread statistics
lon2,lat2 = np.meshgrid(lons,lats)
modelsm = UT.calc_weightedAve(models,lat2)/1e11
obsm = UT.calc_weightedAve(obs,lat2)/1e9

meaens = np.nanmean(modelsm[:,:],axis=0)
maxens = np.nanmax(modelsm[:,:],axis=0)
minens = np.nanmin(modelsm[:,:],axis=0)
spread = maxens - minens

###############################################################################
###############################################################################
###############################################################################
### Create time series
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

###############################################################################
###############################################################################
###############################################################################
### Plot time series of OHC
fig = plt.figure()
ax = plt.subplot(111)

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

ax.fill_between(yearsall,minens[:],maxens[:],facecolor='deepskyblue',alpha=0.25,zorder=1)
plt.plot(yearsall,meaens,color='k',alpha=1,linewidth=0.4,linestyle='--',
         dashes=(1,0.3))

plt.text(yearsall[-1]+1,meaens[-1],r'\textbf{%s}' % modelGCMs[0],
              color='darkblue',fontsize=9,ha='left',va='center')

plt.ylabel(r'\textbf{OHC100 - [ Joules $\times$10$^{11}$]}',fontsize=10,color='dimgrey')
plt.yticks(np.arange(1.00,1.25,0.01),map(str,np.round(np.arange(1.00,1.25,0.01),2)))
plt.xticks(np.arange(1850,2100+1,10),map(str,np.arange(1850,2100+1,10)))
plt.xlim([1979,2100])   
plt.ylim([1.22,1.24])
plt.subplots_adjust(bottom=0.15)
plt.savefig(directoryfigure + 'OHC_TimeSeries_Comparison.png',dpi=300)