"""
Script for calculating the PDO index in each ensemble member

Author     : Zachary M. Labe
Date       : 15 September 2021
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
### Calculate PDO
# mean = UT.calc_weightedAve(models_var,lat2)
# models_varRMmean = models_var - mean[:,:,np.newaxis,np.newaxis]    
models_varRMmean = models_var    
    
latmin = 20 # slices for Pacific Ocean
latmax = 60
lonmin = 110
lonmax = 260
latq = np.where((lats >= latmin) & (lats <= latmax))[0]
lonq = np.where((lons >= lonmin) & (lons <= lonmax))[0]
latslice = lats[latq]
lonslice = lons[lonq]
models_var1 = models_varRMmean[:,:,latq,:]
models_varss = models_var1[:,:,:,lonq]

vectorsst = models_varss.reshape(models_varss.shape[0],models_varss.shape[1],models_varss.shape[2]*models_varss.shape[3])

### Mask out land
mask = ~np.isnan(vectorsst[0,0,:])

### Calculate PDO index for each ensemble member
PDOindexz = np.empty((models_varss.shape[0],models_varss.shape[1]))
PDOindexzWeighted = np.empty((models_varss.shape[0],models_varss.shape[1]))
eof_patternWeighted = []
for i in range(models_varss.shape[0]):
    ensemblesst = vectorsst[i,:,:]
    sstonly = ensemblesst[:,mask]
    
    ### Calculate EOF
    covmat = np.cov(np.transpose(sstonly)) 
    covsparse = sparse.csc_matrix(covmat) 
    
    evals, eof = linalg.eigs(covsparse,k=1) 
    eof = np.real(eof) 
    if eof[300]<0: # same sign of EOF pattern
        eof = -1*eof 
    
    ### Plot EOF pattern
    eofvec = np.empty(models_varss.shape[2]*models_varss.shape[3])
    eofvec[np.argwhere(mask)] = eof
    eofvec[np.argwhere(~mask)] = np.nan
    eofmat = np.reshape(eofvec,(models_varss.shape[2],models_varss.shape[3]))
    plt.figure()
    plt.contourf(eofmat,cmap=cmocean.cm.balance,extend='both')
    
    ### Calculate PDO index from PCs
    PDOindex = np.squeeze(np.matmul(sstonly,eof))
    PDOindexz[i,:] = (PDOindex - np.mean(PDOindex))/np.std(PDOindex)
    
    ###########################################################################
    ### Double check with EOFs package (weight latitudes)
    ensemble = models_varss[i,:,:,:]
    coslat = np.cos(np.deg2rad(latslice))
    wgts = np.sqrt(coslat)[..., np.newaxis]
    solver = Eof(ensemble, weights=wgts)
    
    eof1 = solver.eofsAsCovariance(neofs=1)
    pcsave = solver.pcs(npcs=1, pcscaling=1)[:,0]
    if eof1[0,18,38]<0: # same sign of EOF pattern
        eof1 = -1*eof1 
        pcsave = -1*pcsave
    plt.contourf(eof1[0],cmap=cmocean.cm.balance,extend='both')
    PDOindexzWeighted[i,:] = pcsave
    eof_patternWeighted.append(eof1[0])
    
### Check for differences
if any([np.max(PDOindexz-PDOindexzWeighted)>=0.5,np.min(PDOindexz-PDOindexzWeighted)<=-0.5]):
    print(ValueError('SOMETHING IS WRONG WITH PDO INDEX'))
    sys.exit()
    
### Save PDO index
directoryoutput = '/Users/zlabe/Documents/Research/GmstTrendPrediction/Data/PDO/'
np.savetxt(directoryoutput + 'PDO_CESM2LE_1990-2099.txt',PDOindexzWeighted)

### Create composite of EOF patterns per each ensemble member
eof_patternWeighted = np.asarray(eof_patternWeighted)
meaneof = np.nanmean(eof_patternWeighted,axis=0)

###############################################################################
###############################################################################
### Plot of pdo eof pattern
letters = ["a","b","c","d","e","f","g","h","i","j","k","l","m","n"]
if rm_ensemble_mean == False:
    limit = np.arange(-0.5,0.51,0.02)
    barlim = np.round(np.arange(-0.5,0.51,0.25),2)
elif rm_ensemble_mean == True:
    limit = np.arange(-0.5,0.51,0.02)
    barlim = np.round(np.arange(-0.5,0.51,0.25),2)
cmap = cmocean.cm.balance
label = r'\textbf{SST -- [ PDO COMPOSITE ] -- CESM2-LE}'

fig = plt.figure(figsize=(10,5))
###############################################################################
ax1 = plt.subplot(111)
# m = Basemap(projection='moll',lon_0=0,resolution='l',area_thresh=10000)
# m = Basemap(projection='lcc', resolution='l',lat_1=20,lat_2=60,lat_0=50,lon_0=-107.)
m = Basemap(llcrnrlon=110, llcrnrlat=22, urcrnrlon=260, urcrnrlat=59.5, resolution='l', 
    area_thresh=10000.,projection='merc')
m.drawcoastlines(color='darkgrey',linewidth=0.7)
    
### Variable
varn = meaneof
   
circle = m.drawmapboundary(fill_color='dimgrey',color='dimgray',
                  linewidth=0.7)
circle.set_clip_on(False)

lon2slice,lat2slice = np.meshgrid(lonslice,latslice)
cs1 = m.contourf(lon2slice,lat2slice,varn,limit,extend='both',latlon=True)
cs1.set_cmap(cmap) 
m.fillcontinents(color='dimgrey',lake_color='dimgrey')
        
# ax1.annotate(r'\textbf{[%s]}' % letters[0],xy=(0,0),xytext=(0.93,0.89),
#               textcoords='axes fraction',color='k',fontsize=15,
#               rotation=0,ha='center',va='center')

###############################################################################
cbar_ax1 = fig.add_axes([0.35,0.08,0.3,0.04])                
cbar1 = fig.colorbar(cs1,cax=cbar_ax1,orientation='horizontal',
                    extend='both',extendfrac=0.07,drawedges=False)
cbar1.set_label(label,fontsize=12,color='dimgrey',labelpad=4)  
# cbar1.set_ticks(barlim)
# cbar1.set_ticklabels(list(map(str,barlim)))
cbar1.set_ticks([])
cbar1.set_ticklabels([])
cbar1.ax.tick_params(axis='x', size=.01,labelsize=4)
cbar1.outline.set_edgecolor('dimgrey')

plt.tight_layout()
if rm_ensemble_mean == True:
    plt.savefig(directoryfigure + 'EOFpattern_meanCESMensembles_rmENSEMBLEmean.png',dpi=300)
else:
    plt.savefig(directoryfigure + 'EOFpattern_meanCESMensembles.png',dpi=300)
