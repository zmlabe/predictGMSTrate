"""
Functions for plotting hiatuses using the latest definition (v3)

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
dataset_obs = 'ERA5'
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
directoryfigure = '/Users/zlabe/Desktop/GmstTrendPrediction/'
saveData =  monthlychoice + '_' + variq + '_' + reg_name + '_' + dataset_obs
print('*Filename == < %s >' % saveData) 

### Read data
models,lats,lons = read_primary_dataset(variq,dataset,monthlychoice,numOfEns,
                                        lensalso,randomalso,ravelyearsbinary,
                                        ravelbinary,shuffletype,timeper,
                                        lat_bounds,lon_bounds)
obs,lats_obs,lons_obs = read_obs_dataset(variq,dataset_obs,numOfEns,lensalso,randomalso,ravelyearsbinary,ravelbinary,shuffletype,lat_bounds=lat_bounds,lon_bounds=lon_bounds)

### Calculate global mean temperature
lon2,lat2 = np.meshgrid(lons,lats)
modelsm = UT.calc_weightedAve(models,lat2)
obsm = UT.calc_weightedAve(obs,lat2)

### Call functions
trendlength = 10
AGWstart = 1990
years_newmodel = np.arange(AGWstart,yearsall[-1]+1,1)
years_newobs = np.arange(AGWstart,yearsobs[-1]+1,1)
vv = 0
mo = 0
variq = variables[vv]
monthlychoice = monthlychoiceq[mo]
directoryfigure = '/Users/zlabe/Desktop/GmstTrendPrediction/'
saveData =  monthlychoice + '_' + variq + '_' + reg_name + '_' + dataset_obs
print('*Filename == < %s >' % saveData) 

### Read data for hiatus periods
models = []
modelsm = []
obs = []
obsm = []
SLOPEthreshh_o = []
SLOPEthreshh_m = []
diff_o = []
diff_m = []
yearstrend_obsh = []
linetrend_obsh = []
indexslopeNegative_obsh = []
classes_obsh = []
yearstrend_mh = []
linetrend_mh = []
indexslopeNegative_mh = []
classes_mh = []
count = []
for i in range(len(modelGCMs)):
    dataset = modelGCMs[i]
    modelsq,lats,lons = read_primary_dataset(variq,dataset,monthlychoice,numOfEns,
                                            lensalso,randomalso,ravelyearsbinary,
                                            ravelbinary,shuffletype,timeper,
                                            lat_bounds,lon_bounds)
    obsq,lats,lons = read_obs_dataset(variq,dataset_obs,numOfEns,lensalso,randomalso,ravelyearsbinary,ravelbinary,shuffletype,lat_bounds=lat_bounds,lon_bounds=lon_bounds)
    
    ### Calculate global mean temperature
    lon2,lat2 = np.meshgrid(lons,lats)
    modelsmq = UT.calc_weightedAve(modelsq,lat2)
    obsmq = UT.calc_weightedAve(obsq,lat2)
    
    ### Calculate thresholds for hiatus period
    SLOPEthreshh_oq,diff_oq = HA.calc_thresholdOfTrend(obsmq,trendlength,yearsobs,AGWstart,'hiatus')
    SLOPEthreshh_mq,diff_mq = HA.calc_thresholdOfTrend(modelsmq,trendlength,yearsall,AGWstart,'hiatus')
    
    ### Calculate actual hiatus periods in climate models and observations
    yearstrend_obshq,linetrend_obshq,indexslopeNegative_obshq,classes_obshq = HA.calc_HiatusAcc(obsmq,trendlength,yearsobs,AGWstart,SLOPEthreshh_oq,'hiatus',diff_oq)
    yearstrend_mhq,linetrend_mhq,indexslopeNegative_mhq,classes_mhq = HA.calc_HiatusAcc(modelsmq,trendlength,yearsall,AGWstart,SLOPEthreshh_mq,'hiatus',diff_oq)

    ### County how many hiatus
    countq = len(indexslopeNegative_mhq)

    ### Save for each data set separately
    models.append(modelsq)
    modelsm.append(modelsmq)
    obs.append(obsq)
    obsm.append(obsmq)
    SLOPEthreshh_o.append(SLOPEthreshh_oq)
    SLOPEthreshh_m.append(SLOPEthreshh_mq)
    diff_o.append(diff_oq)
    diff_m.append(diff_mq)
    yearstrend_obsh.append(yearstrend_obshq)
    linetrend_obsh.append(linetrend_obshq)
    indexslopeNegative_obsh.append(indexslopeNegative_obshq)
    classes_obsh.append(classes_obshq)
    yearstrend_mh.append(yearstrend_mhq)
    linetrend_mh.append(linetrend_mhq)
    indexslopeNegative_mh.append(indexslopeNegative_mhq)
    classes_mh.append(classes_mhq)
    count.append(countq)
    
### Check for arrays
models = np.asarray(models)
modelsm = np.asarray(modelsm)
obs = np.asarray(obs).squeeze()
obsm = np.asarray(obsm).squeeze()
SLOPEthreshh_o = np.asarray(SLOPEthreshh_o).squeeze()
SLOPEthreshh_m = np.asarray(SLOPEthreshh_m)
diff_o = np.asarray(diff_o).squeeze()
diff_m = np.asarray(diff_m)
yearstrend_obsh = np.asarray(yearstrend_obsh).squeeze()
linetrend_obsh = np.asarray(linetrend_obsh).squeeze()
indexslopeNegative_obsh = np.asarray(indexslopeNegative_obsh).squeeze()
classes_obsh = np.asarray(classes_obsh).squeeze()
yearstrend_mh = np.asarray(yearstrend_mh)
linetrend_mh = np.asarray(linetrend_mh)
indexslopeNegative_mh = np.asarray(indexslopeNegative_mh)
classes_mh = np.asarray(classes_mh)
count = np.asarray(count)

###############################################################################          
### Calculate ensemble spread statistics
meaens = np.nanmean(modelsm[:,:].squeeze(),axis=0)
maxens = np.nanmax(modelsm[:,:].squeeze(),axis=0)
minens = np.nanmin(modelsm[:,:].squeeze(),axis=0)
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

##############################################################################
##############################################################################
##############################################################################
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

ensmem = 9
plt.plot(yearsall,modelsm.squeeze()[ensmem],color='darkblue',linewidth=2,alpha=1)
plt.plot(yearsall,meaens,color='k',alpha=1,linewidth=0.4,linestyle='--',
         dashes=(1,0.3))
for hi in range(yearstrend_mh.squeeze().shape[1]):
    if linetrend_mh.squeeze()[ensmem,hi,0] <= SLOPEthreshh_m[0][hi]*diff_o:
        plt.plot(yearstrend_mh.squeeze()[ensmem,hi],linetrend_mh.squeeze()[ensmem,hi,0]*yearstrend_mh.squeeze()[0,hi]+linetrend_mh.squeeze()[ensmem,hi,1],
                  color='r',linewidth=1,zorder=5,clip_on=False)

plt.text(yearsall[-1]+1,meaens[-1],r'\textbf{%s}' % modelGCMs[0],
              color='darkblue',fontsize=9,ha='left',va='center')

plt.ylabel(r'\textbf{GMST [$^{\circ}$C]}',fontsize=10,color='dimgrey')
plt.yticks(np.arange(0,21,0.5),map(str,np.round(np.arange(0,21,0.5),2)))
plt.xticks(np.arange(1850,2100+1,25),map(str,np.arange(1850,2100+1,25)))
plt.xlim([1990,2100])   
plt.ylim([14,19])
plt.subplots_adjust(bottom=0.15)
plt.savefig(directoryfigure + 'TimeSeries_MeanGlobalTemperature_%s_withHiatus-v3.png' % modelGCMs[0],
             dpi=600)

###############################################################################
###############################################################################
###############################################################################
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

plt.plot(yearsobs,obsm,color='darkblue',linewidth=4,alpha=1,clip_on=True)
for hi in range(len(linetrend_obsh)):
    if linetrend_obsh[hi,0] <= SLOPEthreshh_o:
        plt.plot(yearstrend_obsh[hi,:],linetrend_obsh[hi,0]*yearstrend_obsh[hi,:]+linetrend_obsh[hi,1],
             color='r',linewidth=1)
    
plt.ylabel(r'\textbf{GMST [$^{\circ}$C]}',fontsize=10,color='dimgrey')
plt.yticks(np.arange(0,21,0.5),map(str,np.round(np.arange(0,21,0.5),2)))
plt.xticks(np.arange(1850,2100+1,10),map(str,np.arange(1850,2100+1,10)))
plt.xlim([1990,2020])   
plt.ylim([14,15])
plt.subplots_adjust(bottom=0.15)
plt.savefig(directoryfigure + 'TimeSeries_MeanGlobalTemperature_ERA5_withHiatus-v3.png',
            dpi=300)