"""
Functions calculate hiatus definition
 
Notes
-----
    Author : Zachary Labe
    Date   : 17 August 2021
    
Usage
-----
    [1] calc_thresholdOfTrend(data,trendlength,years,AGWstart)
    [2] calc_Hiatus(data,trendlength,years,AGWstart,SLOPEthresh)
"""
def calc_thresholdOfTrend(data,trendlength,years,AGWstart): 
    """
    Function calculates threshold for trend analysis of hiatus

    Parameters
    ----------
    data : numpy array
        array of climate model or reanalysis data
        
    Returns
    -------
    data : n-d numpy array
        data from selected data set
    trendlength : integer
        length of trend periods to calculate
    years : 1d array
        Original years of input data
    AGWstart : integer
        Start of data to calculate trends over

    Usage
    -----
    SLOPEthresh = calc_thresholdOfTrend(data,trendlength,years,AGWstart)
    """
    print('\n>>>>>>>>>> Using calc_thresholdOfTrend function!')
    
    ### Import modules
    import numpy as np
    import sys
    
    ### Pick start of forced climate change
    yrq = np.where(years[:] >= AGWstart)[0]
    data = data[yrq]
    yearsnew = years[yrq]
    print('Years-Trend ---->\n',yearsnew)
    
    if data.ndim == 1:    
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
            
        ### Slopes
        slope = linetrend[:,0]
        stdslope = np.nanstd(slope)
        meantrend = np.nanmean(slope)
        
        print('\nHIATUS is %s years long!' % trendlength)
        print('-- Number of years is',years.shape[0],'and number of trends is',slope.shape[0],'--')
        
        SLOPEthresh = meantrend - stdslope
    else:
        print(ValueError('WRONG DIMENSIONS OF OBS!'))
        sys.exit()
        
    print('>>>>>>>>>> Ending calc_thresholdOfTrend function!')
    return SLOPEthresh

def calc_Hiatus(data,trendlength,years,AGWstart,SLOPEthresh):
    """
    Function calculates threshold for trend analysis of hiatus

    Parameters
    ----------
    data : numpy array
        array of climate model or reanalysis data
        
    Returns
    -------
    data : n-d numpy array
        data from selected data set
    trendlength : integer
        length of trend periods to calculate
    years : 1d array
        Original years of input data
    AGWstart : integer
        Start of data to calculate trends over

    Usage
    -----
    SLOPEthresh = calc_thresholdOfTrend(data,trendlength,years,AGWstart)
    """
    print('\n>>>>>>>>>> Using calc_Hiatus function!')
    hiatusSLOPE = SLOPEthresh
    
    if data.ndim == 2:
        yrq = np.where(years[:] >= AGWstart)[0]
        data = data[:,yrq]
        yearsnew = years[yrq]
        print('Years-Trend ---->\n',yearsnew)
        
        ens = len(data)
        
        ### Calculate trend periods
        yearstrend = np.empty((ens,len(yearsnew)-trendlength+1,trendlength))
        datatrend = np.empty((ens,len(yearsnew)-trendlength+1,trendlength))
        for e in range(ens):
            for hi in range(len(yearsnew)-trendlength+1):
                yearstrend[e,hi,:] = np.arange(yearsnew[hi],yearsnew[hi]+trendlength,1)
                datatrend[e,hi,:] = data[e,hi:hi+trendlength]        
                 
        ### Calculate trend lines
        linetrend = np.empty((ens,len(years)-trendlength+1,2))
        for e in range(ens):
            for hi in range(len(yearsnew)-trendlength+1):
                linetrend[e,hi,:] = np.polyfit(yearstrend[e,hi],datatrend[e,hi],1)
            
        ### Count number of hiatus periods
        slope = linetrend[:,:,0]
        indexslopeNegative = []
        for e in range(ens):
            indexslopeNegativeq = np.where((slope[e,:] <= hiatusSLOPE))[0]
            if len(indexslopeNegativeq) == 0:
                indexslopeNegative.append([np.nan])
            else:
                indexslopeNegative.append(indexslopeNegativeq)
          
        ### Calculate classes
        classes = np.zeros((data.shape))
        for e in range(ens):
            indexFirstHiatus = []
            for i in range(len(indexslopeNegative[e])):
                if i == 0:
                    saveFirstHiatusYR = indexslopeNegative[e][i]
                    indexFirstHiatus.append(saveFirstHiatusYR)
                elif indexslopeNegative[e][i]-1 != indexslopeNegative[e][i-1]:
                    saveFirstHiatusYR = indexslopeNegative[e][i]
                    indexFirstHiatus.append(saveFirstHiatusYR)
                
            classes[e,indexFirstHiatus] = 1
                
    elif data.ndim == 1:    
        yrq = np.where(years[:] >= AGWstart)[0]
        data = data[yrq]
        yearsnew = years[yrq]
        print('Years-Trend ---->\n',yearsnew)
      
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
            
        ### Count number of hiatus periods
        slope = linetrend[:,0]
        indexslopeNegative = np.where((slope[:] <= hiatusSLOPE))[0]
        print('INDEX OF HIATUS---->',indexslopeNegative)
        
        ### Calculate classes
        indexFirstHiatus = []
        for i in range(len(indexslopeNegative)):
            if i == 0:
                saveFirstHiatusYR = indexslopeNegative[i]
                indexFirstHiatus.append(saveFirstHiatusYR)
            elif indexslopeNegative[i]-1 != indexslopeNegative[i-1]:
                saveFirstHiatusYR = indexslopeNegative[i]
                indexFirstHiatus.append(saveFirstHiatusYR)
        classes = np.zeros((len(yearsnew)))
        classes[indexFirstHiatus] = 1
     
        print('\n>>>>>>>>>> Ending calc_Hiatus function!')                         
    return yearstrend,linetrend,indexslopeNegative,classes









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
modelGCMs = ['CESM2LE']
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
obs,lats,lons = read_obs_dataset(variq,dataset_obs,numOfEns,lensalso,randomalso,ravelyearsbinary,ravelbinary,shuffletype,lat_bounds=lat_bounds,lon_bounds=lon_bounds)

### Calculate global mean temperature
lon2,lat2 = np.meshgrid(lons,lats)
modelsm = UT.calc_weightedAve(models,lat2)
obsm = UT.calc_weightedAve(obs,lat2)

trendlength = 10
AGWstart = 1990

SLOPEthresh = calc_thresholdOfTrend(obsm,trendlength,yearsobs,AGWstart)
yearstrend_obs,linetrend_obs,indexslopeNegative_obs,classes_obs = calc_Hiatus(obsm,trendlength,yearsobs,AGWstart,SLOPEthresh)
yearstrend_m,linetrend_m,indexslopeNegative_m,classes_m = calc_Hiatus(modelsm,trendlength,yearsall,AGWstart,SLOPEthresh)