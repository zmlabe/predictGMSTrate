"""
Plot heat map of IPO index (validation data)

Author     : Zachary M. Labe
Date       : 17 September 2021
Version    : 2 (mostly for testing)
"""

### Import packages
import sys
import matplotlib.pyplot as plt
import numpy as np
import calc_Utilities as UT
import palettable.cubehelix as cm
import cmocean as cmocean
import matplotlib.colors as c

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
monthlychoiceq = ['annual']
variables = ['T2M']
vari_predict = ['OHC100']
if vari_predict[0][:3] == 'OHC':
    obs_predict = 'OHC'
else:
    obs_predict = 'ERA5'
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
lat_bounds,lon_bounds = UT.regions(reg_name)
###############################################################################
###############################################################################
ravelyearsbinary = False
ravelbinary = False
lensalso = True
###############################################################################
###############################################################################
### Remove ensemble mean 
rm_ensemble_mean = True
###############################################################################
###############################################################################
### Call functions
trendlength = 10
AGWstart = 1990
years_newmodel = np.arange(AGWstart,yearsall[-1]+1,1)
years_newobs = np.arange(AGWstart,yearsobs[-1]+1,1)
vv = 0
mo = 0
variq = variables[vv]
monthlychoice = monthlychoiceq[mo]
directoryfigure = '/Users/zlabe/Desktop/GmstTrendPrediction/ANN_v2/PDO/'
saveData =  monthlychoice + '_' + variq + '_' + reg_name + '_' + dataset_obs
print('*Filename == < %s >' % saveData) 

### Parameters from ANN
if rm_ensemble_mean == False:
    variq = 'T2M'
    fac = 0.7
    random_segment_seed = int(np.genfromtxt('/Users/zlabe/Documents/Research/GmstTrendPrediction/Data/SelectedSegmentSeed.txt',unpack=True))
    random_network_seed = 87750
    hidden = [20,20]
    n_epochs = 500
    batch_size = 128
    lr_here = 0.001
    ridgePenalty = 0.05
    actFun = 'relu'
    fractWeight = 0.5
    yearsall = np.arange(1990,2090+1,1)
elif rm_ensemble_mean == True:
    variq = 'T2M'
    fac = 0.7
    random_segment_seed = int(np.genfromtxt('/Users/zlabe/Documents/Research/GmstTrendPrediction/Data/SelectedSegmentSeed.txt',unpack=True))
    random_network_seed = 87750
    hidden = [30,30]
    n_epochs = 500
    batch_size = 128
    lr_here = 0.001
    ridgePenalty = 0.5
    actFun = 'relu'
    fractWeight = 0.5
    yearsall = np.arange(1990,2090+1,1)
else:
    print(ValueError('SOMETHING IS WRONG WITH DATA PROCESSING!'))
    sys.exit()

### Naming conventions for files
directorymodel = '/Users/zlabe/Documents/Research/GmstTrendPrediction/SavedModels/'
savename = 'ANNv2_'+vari_predict[0]+'_hiatus_' + actFun + '_L2_'+ str(ridgePenalty)+ '_LR_' + str(lr_here)+ '_Batch'+ str(batch_size)+ '_Iters' + str(n_epochs) + '_' + str(len(hidden)) + 'x' + str(hidden[0]) + '_SegSeed' + str(random_segment_seed) + '_NetSeed'+ str(random_network_seed) 
if(rm_ensemble_mean==True):
    savename = savename + '_EnsembleMeanRemoved'   
    
### Directories to save files
directorydata = '/Users/zlabe/Documents/Research/GmstTrendPrediction/Data/'

###############################################################################
###############################################################################
###############################################################################
### Read in data for testing predictions and actual hiatuses
validationindices = np.asarray(np.genfromtxt(directorydata + 'validationEnsIndices_' + savename + '.txt'),dtype=int)
actual_validation = np.genfromtxt(directorydata + 'validationTrueLabels_' + savename + '.txt')
predict_validation = np.genfromtxt(directorydata + 'validationPredictedLabels_' + savename+ '.txt')

### Reshape arrays for [ensemble,year]
act_re = np.swapaxes(actual_validation.reshape(validationindices.shape[0],1,yearsall.shape[0]),0,1).squeeze()
pre_re = np.swapaxes(predict_validation.reshape(validationindices.shape[0],1,yearsall.shape[0]),0,1).squeeze()

act_nan_re = act_re.copy()
act_nan_re[np.where(act_nan_re == 0.)] = np.nan
pre_nan_re = pre_re.copy()
pre_nan_re[np.where(pre_nan_re == 0.)]= np.nan

### Read in IPO index
IPO = np.genfromtxt(directorydata + '/IPO/IPO_CESM2LE_1990-2099.txt',unpack=True)
IPOvalidation = IPO.transpose()[validationindices,:yearsall.shape[0]]

###############################################################################
###############################################################################
###############################################################################
###############################################################################                      
### Call parameters
plt.rc('text',usetex=True)
plt.rc('font',**{'family':'sans-serif','sans-serif':['Avant Garde']}) 

### Plot first meshgrid
fig = plt.figure(figsize=(10,6))
ax = plt.subplot(111)

ax.spines['top'].set_color('none')
ax.spines['right'].set_color('none')
ax.spines['bottom'].set_color('none')
ax.spines['left'].set_color('none')
ax.get_xaxis().set_tick_params(direction='out', width=2,length=3,
            color='dimgrey')
ax.get_yaxis().set_tick_params(direction='out', width=2,length=3,
            color='dimgrey')

plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom='on',      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom='on')
plt.tick_params(
    axis='y',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    left='on',      # ticks along the bottom edge are off
    right=False,         # ticks along the top edge are off
    labelleft='on')

csm=plt.get_cmap(cmocean.cm.balance)
norm = c.BoundaryNorm(np.arange(-3,3.1,0.1),csm.N)

cs = plt.pcolormesh(IPOvalidation,shading='faceted',edgecolor='w',
                    linewidth=0.05,vmin=-3,vmax=3,norm=norm,cmap=csm)

ensembles = np.arange(1,6+1,1)
plt.yticks(np.arange(0.5,6.5,1),ensembles,ha='right',va='center',color='k',size=6)
yax = ax.get_yaxis()
yax.set_tick_params(pad=2)
plt.xticks(np.arange(0.5,101.5,10),map(str,np.arange(1990,2091,10)),
           color='k',size=6)
plt.xlim([0,101])

saveIPO = []
predictedsaveIPO = []
predictedsaveIPOall = []
for i in range(act_nan_re.shape[0]):
    for j in range(act_nan_re.shape[1]):
        if pre_nan_re[i,j] == 1:
            cc = 'dimgrey'
            plt.text(j+0.56,i+0.48,r'\textbf{H}',fontsize=9,
                color=cc,va='center',ha='center')
            predictedsaveIPOall.append(IPOvalidation[i,j])
        if np.isnan(act_nan_re[i,j]) == False:
            if pre_nan_re[i,j] == 1:
                cc ='gold'
                predictedsaveIPO.append(IPOvalidation[i,j])
            else:
                cc = 'k'        
            saveIPO.append(IPOvalidation[i,j])
            plt.text(j+0.56,i+0.48,r'\textbf{H}',fontsize=9,
                color=cc,va='center',ha='center')
saveIPO = np.asarray(saveIPO)
meanIPOsave = np.round(np.nanmean(saveIPO),2)
predictedsaveIPO = np.asarray(predictedsaveIPO)
meanIPOpredictedsave = np.round(np.nanmean(predictedsaveIPO),2)
predictedsaveIPOall = np.asarray(predictedsaveIPOall)
meanIPOpredictedsaveall = np.round(np.nanmean(predictedsaveIPOall),2)
print(meanIPOsave,meanIPOpredictedsave,meanIPOpredictedsaveall)
                 
cbar = plt.colorbar(cs,orientation='horizontal',aspect=80,pad=0.12,
                    extend='both')
cbar.set_ticks(np.arange(-3,3.1,1))
cbar.set_ticklabels(np.arange(-3,3.1,1),map(str,np.round(np.arange(-3,3.1,1),2)))  
cbar.ax.tick_params(axis='x', size=.001,labelsize=7)
cbar.outline.set_edgecolor('dimgrey')
cbar.set_label(r'\textbf{ANNUAL IPO INDEX [CESM2-LE]}',
                color='k',labelpad=10,fontsize=23)

plt.ylabel(r'\textbf{Ensemble Member}',color='k',fontsize=12)

plt.tight_layout()
plt.savefig(directoryfigure + 'IPO_heatmap_hiatus_v2-Validation.png',dpi=300)