"""
Script plots the correlation between the decadal trend and IPO index

Author     : Zachary M. Labe
Date       : 19 October 2021
Version    : 2
"""

### Import packages
import sys
import matplotlib.pyplot as plt
import numpy as np
import calc_Hiatus_v4 as HA
import calc_Utilities as UT
import calc_dataFunctions as df
import calc_Stats as dSS
import scipy.stats as sts

### Plotting defaults 
plt.rc('text',usetex=True)
plt.rc('font',**{'family':'sans-serif','sans-serif':['Avant Garde']})

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
### Data preliminaries 
modelGCMs = ['CESM2LE']
dataset_obs = 'ERA5'
allDataLabels = modelGCMs
letters = ["a","b","c","d","e","f","g","h","i","j","k","l","m"]
datasetsingle = ['CESM2le']
monthlychoiceq = ['annual']
variables = ['T2M']
reg_name = 'SMILEGlobe'
level = 'surface'
ens1 = np.arange(1,10+1,1)
ens2 = np.arange(21,50+1,1)
ens = np.append(ens1,ens2)
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
baseline = np.arange(1981,2010+1,1)
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
# ###############################################################################
# ###############################################################################
# ###############################################################################
# ###############################################################################
# ### Read in model and observational/reanalysis data
# def read_primary_dataset(variq,dataset,monthlychoice,numOfEns,lensalso,randomalso,ravelyearsbinary,ravelbinary,shuffletype,timeper,lat_bounds=lat_bounds,lon_bounds=lon_bounds):
#     data,lats,lons = df.readFiles(variq,dataset,monthlychoice,numOfEns,lensalso,randomalso,ravelyearsbinary,ravelbinary,shuffletype,timeper)
#     datar,lats,lons = df.getRegion(data,lats,lons,lat_bounds,lon_bounds)
#     print('\nOur dataset: ',dataset,' is shaped',data.shape)
#     return datar,lats,lons
  
# def read_obs_dataset(variq,dataset_obs,numOfEns,lensalso,randomalso,ravelyearsbinary,ravelbinary,shuffletype,lat_bounds=lat_bounds,lon_bounds=lon_bounds):
#     data_obs,lats_obs,lons_obs = df.readFiles(variq,dataset_obs,monthlychoice,numOfEns,lensalso,randomalso,ravelyearsbinary,ravelbinary,shuffletype,timeper)
#     data_obs,lats_obs,lons_obs = df.getRegion(data_obs,lats_obs,lons_obs,lat_bounds,lon_bounds)
    
#     print('our OBS dataset: ',dataset_obs,' is shaped',data_obs.shape)
#     return data_obs,lats_obs,lons_obs

# ### Call functions
# vv = 0
# mo = 0
# variq = variables[vv]
# monthlychoice = monthlychoiceq[mo]
# directoryfigure = '/Users/zlabe/Desktop/GmstTrendPrediction/ANN_v2/PDO/'
# saveData =  monthlychoice + '_' + variq + '_' + reg_name + '_' + dataset_obs
# print('*Filename == < %s >' % saveData) 

# ### Read data
# models,lats,lons = read_primary_dataset(variq,dataset,monthlychoice,numOfEns,
#                                         lensalso,randomalso,ravelyearsbinary,
#                                         ravelbinary,shuffletype,timeper,
#                                         lat_bounds,lon_bounds)
# obs,lats,lons = read_obs_dataset(variq,dataset_obs,numOfEns,lensalso,randomalso,ravelyearsbinary,ravelbinary,shuffletype,lat_bounds=lat_bounds,lon_bounds=lon_bounds)

# models,obs = dSS.calculate_anomalies(models,obs,lats,lons,baseline,yearsall,yearsobs)

# ### Calculate global mean temperature
# lon2,lat2 = np.meshgrid(lons,lats)
# modelsm = UT.calc_weightedAve(models,lat2)
# obsm = UT.calc_weightedAve(obs,lat2)
# meaens = np.nanmean(modelsm[:,:].squeeze(),axis=0)
# maxens = np.nanmax(modelsm[:,:].squeeze(),axis=0)
# minens = np.nanmin(modelsm[:,:].squeeze(),axis=0)
# spread = maxens - minens

# trendlength = 10
# AGWstart = 1990

# SLOPEthreshh_o,diff_o = HA.calc_thresholdOfTrend(obsm,trendlength,yearsobs,AGWstart,'hiatus')
# SLOPEthreshh_m,diff_m = HA.calc_thresholdOfTrend(modelsm,trendlength,yearsall,AGWstart,'hiatus')
# yearstrend_obsh,linetrend_obsh,indexslopeNegative_obsh,classes_obsh = HA.calc_HiatusAcc(obsm,trendlength,yearsobs,AGWstart,SLOPEthreshh_o,'hiatus',diff_o)
# yearstrend_mh,linetrend_mh,indexslopeNegative_mh,classes_mh = HA.calc_HiatusAcc(modelsm,trendlength,yearsall,AGWstart,SLOPEthreshh_m,'hiatus',diff_o)

###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
### Naming conventions for files
yearsnew = np.arange(1990,2090+1,1)
directorymodel = '/Users/zlabe/Documents/Research/GmstTrendPrediction/SavedModels/'
savename = 'ANNv2_OHC100_hiatus_relu_L2_0.5_LR_0.001_Batch128_Iters500_2x30_SegSeed24120_NetSeed87750' 
rm_ensemble_mean = True
if(rm_ensemble_mean==True):
    savename = savename + '_EnsembleMeanRemoved'   
    
### Directories to save files
directorydata = '/Users/zlabe/Documents/Research/GmstTrendPrediction/Data/'

### Read in data for testing predictions and actual hiatuses
testindices = np.asarray(np.genfromtxt(directorydata + 'testingEnsIndices_' + savename + '.txt'),dtype=int)
actual_test = np.genfromtxt(directorydata + 'testingTrueLabels_' + savename + '.txt')

### Reshape arrays for [ensemble,year]
act_retest = np.swapaxes(actual_test.reshape(testindices.shape[0],1,yearsnew.shape[0]),0,1).squeeze()

### Read in IPO index
# IPO = np.genfromtxt(directorydata + '/PDO/PDO_CESM2LE_1990-2099.txt',unpack=True)
IPO = np.genfromtxt(directorydata + '/IPO/IPO_CESM2LE_1990-2099.txt',unpack=True)
IPOtest = IPO.transpose()[testindices,:yearsnew.shape[0]]

### Picking ensembles for testing
slope = linetrend_mh[:,:,0]
testens_slope = slope[testindices,:]

### Correlation
r,p = sts.pearsonr(IPOtest.ravel(),testens_slope.ravel())

### Plot on hiatus events
IPO_hiatusevents = IPOtest * act_retest
IPO_hiatusevents[np.where(IPO_hiatusevents==0.)] = np.nan
    
###############################################################################
###############################################################################
###############################################################################
### Create arrays for plotting
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
# ax.yaxis.grid(zorder=1,color='dimgrey',alpha=0.35,clip_on=False)
ax.xaxis.grid(zorder=1,color='dimgrey',alpha=0.35,clip_on=False)

plt.axvline(x=0,color='dimgrey',linestyle='-',linewidth=2,clip_on=False)

plt.scatter(IPOtest.ravel(),testens_slope.ravel(),s=36,color='dimgrey',clip_on=False,
            zorder=3,edgecolor='k',linewidth=0.5,alpha=0.5,label=r'\textbf{Decadal Warming}')
plt.scatter(IPO_hiatusevents.ravel(),testens_slope.ravel(),s=36,color='maroon',clip_on=False,
            zorder=3,edgecolor='darkred',linewidth=0.5,alpha=0.5,label=r'\textbf{Slowdown}')
    
plt.xticks(np.arange(-5,6,1),map(str,np.arange(-5,6,1)),size=6)
plt.yticks(np.arange(-0.1,0.15,0.05),map(str,np.round(np.arange(-0.1,0.15,0.05),2)),size=6)
plt.xlim([-4,4])   
plt.ylim([-0.05,0.1])
plt.ylabel(r'\textbf{10-Year Trends in GMST [$^{\circ}$C/yr]}',color='k',
           fontsize=10)
plt.xlabel(r'\textbf{Unfiltered IPO Index}',color='k',
           fontsize=10)

plt.text(3.95,-0.049,r'\textbf{R$^{2}$=%s}' % (np.round(r**2,2)),
         color='k',fontsize=12,ha='right')

leg = plt.legend(shadow=False,fontsize=10,loc='upper center',
                 bbox_to_anchor=(0.5,1.15),
                 fancybox=True,ncol=3,frameon=False,
                 handlelength=0)
for h, t in zip(leg.legendHandles, leg.get_texts()):
    t.set_color(h.get_facecolor()[0])

if rm_ensemble_mean == True:
    plt.savefig(directoryfigure + 'CorrelationIPO-Hiatus_EDA-v2_rmENSEMBLEmean.png',dpi=300)
else:
    plt.savefig(directoryfigure + 'CorrelationIPO-Hiatus_EDA-v2.png',dpi=300)