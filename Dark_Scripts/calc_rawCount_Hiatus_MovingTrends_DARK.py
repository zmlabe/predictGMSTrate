"""
Functions for calculating hiatus periods in GMST

Author     : Zachary M. Labe
Date       : 5 August 2021
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
yearsall = np.arange(1980+window,2099+1,1)
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

# ### Call functions
# vv = 0
# mo = 0
# variq = variables[vv]
# monthlychoice = monthlychoiceq[mo]
# directoryfigure = '/Users/zlabe/Documents/Projects/predictGMSTrate/Dark_Figures/'
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

def calc_Hiatus(data,trendlength,years,AGWstart,SLOPEthresh):
    hiatusSLOPE = SLOPEthresh
    
    if data.ndim == 2:
        ens = len(data)
        
        ### Calculate trend periods
        yearstrend = np.empty((ens,len(years)-trendlength,trendlength))
        datatrend = np.empty((ens,len(years)-trendlength,trendlength))
        for e in range(ens):
            for hi in range(len(years)-trendlength):
                yearstrend[e,hi,:] = np.arange(years[hi],years[hi]+trendlength,1)
                datatrend[e,hi,:] = data[e,hi:hi+trendlength]
            
        ### Calculate trend lines
        linetrend = np.empty((ens,len(years)-trendlength,2))
        for e in range(ens):
            for hi in range(len(years)-trendlength):
                linetrend[e,hi,:] = np.polyfit(yearstrend[e,hi],datatrend[e,hi],1)
        
        ### Test plot trend lines
        fig = plt.figure()        
        for hi in range(len(years)-trendlength):
            plt.plot(yearstrend[0,hi],linetrend[0,hi,0]*yearstrend[0,hi]+linetrend[0,hi,1],
                     color='darkblue',linewidth=0.8)
            
        ### Pick start of forced climate change
        yrq = np.where(years[:-trendlength] >= AGWstart)[0]
        linetrend = linetrend[:,yrq,:]
        yearsnew = years[yrq]
            
        ### Count number of hiatus periods
        slope = linetrend[:,:,0]
        indexslopeNegative = []
        for e in range(ens):
            indexslopeNegativeq = np.where((slope[e,:] <= hiatusSLOPE))[0]
            if len(indexslopeNegativeq) == 0:
                indexslopeNegative.append([np.nan])
            else:
                indexslopeNegative.append(indexslopeNegativeq)
                
    elif data.ndim == 1:    
        ### Calculate trend periods
        yearstrend = np.empty((len(years)-trendlength+1,trendlength))
        datatrend = np.empty((len(years)-trendlength+1,trendlength))
        for hi in range(len(years)-trendlength+1):
            yearstrend[hi,:] = np.arange(years[hi],years[hi]+trendlength,1)
            datatrend[hi,:] = data[hi:hi+trendlength]
            
        ### Calculate trend lines    
        linetrend = np.empty((len(years)-trendlength+1,2))
        for hi in range(len(years)-trendlength+1):
            linetrend[hi,:] = np.polyfit(yearstrend[hi],datatrend[hi],1)
         
        ### Test plot trend lines
        fig = plt.figure()
        for hi in range(len(years)-trendlength):
            plt.plot(yearstrend[hi],linetrend[hi,0]*yearstrend[hi]+linetrend[hi,1],
                     color='darkred',linewidth=0.8)
            
        ### Pick start of forced climate change
        yrq = np.where(years[:-trendlength+1] >= AGWstart)[0]
        linetrend = linetrend[yrq,:]
        yearsnew = years[yrq]
            
        ### Count number of hiatus periods
        slope = linetrend[:,0]
        indexslopeNegative = np.where((slope[:] <= hiatusSLOPE))[0]
              
    return yearstrend,linetrend,indexslopeNegative

def calc_TrendsObs(data,trendlength,years,AGWstart): 
    ### Pick start of forced climate change
    yrq = np.where(years[:-trendlength] >= AGWstart)[0]
    data = data[yrq]
    years = years[yrq]
    
    if data.ndim == 1:    
    ### Calculate trend periods
        yearstrend = np.empty((len(years)-trendlength,trendlength))
        datatrend = np.empty((len(years)-trendlength,trendlength))
        for hi in range(len(years)-trendlength):
            yearstrend[hi,:] = np.arange(years[hi],years[hi]+trendlength,1)
            datatrend[hi,:] = data[hi:hi+trendlength]
            
        ### Calculate trend lines    
        linetrend = np.empty((len(years)-trendlength,2))
        for hi in range(len(years)-trendlength):
            linetrend[hi,:] = np.polyfit(yearstrend[hi],datatrend[hi],1)
            
        ### Slopes
        slope = linetrend[:,0]
        stdslope = np.nanstd(slope)
        meantrend = np.nanmean(slope)
        
        SLOPEthresh = meantrend - stdslope
        return SLOPEthresh

# hiatus = 10
# AGWstart = 1980
# SLOPEthresh = calc_TrendsObs(obsm,hiatus,yearsobs,AGWstart)
years_obs,linetrend_obs,indexneg_obs = calc_Hiatus(obsm,hiatus,yearsobs,AGWstart,SLOPEthresh)
# years_model,linetrend_model,indexneg_model = calc_Hiatus(modelsm,hiatus,yearsall,AGWstart,SLOPEthresh)

# numberOfHiatus = np.empty((len(indexneg_model)))
# for i in range(len(indexneg_model)):
#     numberOfHiatus[i] = len(indexneg_model[i])
# print('\n',numberOfHiatus)

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


ax.fill_between(yearsall,minens[:],maxens[:],facecolor='w',alpha=0.5,zorder=1,clip_on=False)

ensmem = 7
plt.plot(yearsall,modelsm[ensmem],color='gold',linewidth=2,alpha=1)
for hi in range(len(yearsall)-hiatus):
    plt.plot(years_model[ensmem,hi],linetrend_model[ensmem,hi,0]*years_model[0,hi]+linetrend_model[ensmem,hi,1],
             color='deepskyblue',linewidth=0.3,zorder=5,clip_on=False)

plt.text(yearsall[-1]+1,meaens[-1],r'\textbf{%s}' % modelGCMs[0],
              color='gold',fontsize=9,ha='left',va='center')

plt.ylabel(r'\textbf{GMST [$^{\circ}$C]}',fontsize=10,color='darkgrey')
plt.yticks(np.arange(0,21,0.5),map(str,np.round(np.arange(0,21,0.5),2)))
plt.xticks(np.arange(1850,2100+1,25),map(str,np.arange(1850,2100+1,25)))
plt.xlim([1975,2100])   
plt.ylim([13,19])
plt.subplots_adjust(bottom=0.15)
plt.savefig(directoryfigure + 'TimeSeries_MeanGlobalTemperature_%sonly_HIATUStrends_DARK.png'  % modelGCMs[0],
            dpi=600)

###############################################################################
###############################################################################
###############################################################################
fig = plt.figure()
ax = plt.subplot(111)

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

ax.fill_between(yearsall,minens[:],maxens[:],facecolor='w',alpha=0.5,zorder=1,clip_on=False)

ensmem = 7
plt.plot(yearsall,modelsm[ensmem],color='gold',linewidth=2,alpha=1)
for hi in range(len(yearsall)-hiatus):
    if linetrend_model[ensmem,hi,0] <= SLOPEthresh:
        plt.plot(years_model[ensmem,hi],linetrend_model[ensmem,hi,0]*years_model[0,hi]+linetrend_model[ensmem,hi,1],
                 color='blue',linewidth=1,zorder=5,clip_on=False)

plt.text(yearsall[-1]+1,meaens[-1],r'\textbf{%s}' % modelGCMs[0],
              color='gold',fontsize=9,ha='left',va='center')

plt.ylabel(r'\textbf{GMST [$^{\circ}$C]}',fontsize=10,color='darkgrey')
plt.yticks(np.arange(0,21,0.5),map(str,np.round(np.arange(0,21,0.5),2)))
plt.xticks(np.arange(1850,2100+1,25),map(str,np.arange(1850,2100+1,25)))
plt.xlim([1975,2100])   
plt.ylim([13,19])
plt.subplots_adjust(bottom=0.15)
plt.savefig(directoryfigure + 'TimeSeries_MeanGlobalTemperature_%sonly_HIATUS-ONLY-trends_DARK.png' % modelGCMs[0],
            dpi=600)

###############################################################################
###############################################################################
###############################################################################
fig = plt.figure()
ax = plt.subplot(111)

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

plt.plot(yearsobs,obsm,color='darkblue',linewidth=2,alpha=1)
for hi in range(len(yearsobs)-hiatus-1):
    plt.plot(years_obs[hi],linetrend_obs[hi,0]*years_obs[hi]+linetrend_obs[hi,1],
             color='darkred',linewidth=0.8)
    
plt.ylabel(r'\textbf{GMST [$^{\circ}$C]}',fontsize=10,color='darkgrey')
plt.yticks(np.arange(0,21,0.5),map(str,np.round(np.arange(0,21,0.5),2)))
plt.xticks(np.arange(1850,2100+1,25),map(str,np.arange(1850,2100+1,25)))
plt.xlim([1975,2025])   
plt.ylim([13.5,15])
plt.subplots_adjust(bottom=0.15)
plt.savefig(directoryfigure + 'TimeSeries_MeanGlobalTemperature_ERA5_HIATUStrends_DARK.png',
            dpi=600)

###############################################################################
###############################################################################
###############################################################################
fig = plt.figure()
ax = plt.subplot(111)

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

plt.plot(yearsobs,obsm,color='crimson',linewidth=3,alpha=1,clip_on=False)
for hi in range(len(yearsobs)-hiatus):
    if any([hi==31,hi==22]):
        plt.plot(years_obs[hi],linetrend_obs[hi,0]*years_obs[hi]+linetrend_obs[hi,1],
             color='deepskyblue',linewidth=4,linestyle='--',dashes=(1,0.3),clip_on=False)
    
plt.ylabel(r'\textbf{GMST [$^{\circ}$C]}',fontsize=10,color='darkgrey')
plt.yticks(np.arange(0,21,0.1),map(str,np.round(np.arange(0,21,0.1),2)))
plt.xticks(np.arange(1950,2100+1,5),map(str,np.arange(1950,2100+1,5)))
plt.xlim([1979,2020])   
plt.ylim([13.8,14.8])
plt.subplots_adjust(bottom=0.15)
plt.savefig(directoryfigure + 'TimeSeries_MeanGlobalTemperature_ERA5_HIATUS-ONLY-trends_DARK.png',
            dpi=600)