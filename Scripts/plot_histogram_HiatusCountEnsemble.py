"""
Create plot of histogram showing the number of hiatuses for each ensemble

Author     : Zachary M. Labe
Date       : 30 September 2021
Version    : 2 (mostly for testing)
"""

### Import packages
import sys
import matplotlib.pyplot as plt
import matplotlib.colors as c
import numpy as np
from netCDF4 import Dataset
import scipy.stats as stats
import palettable.cubehelix as cm
import cmocean as cmocean
import cmasher as cmr
import calc_Utilities as UT
import calc_dataFunctions as df
import calc_Stats as dSS
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score,confusion_matrix,precision_recall_fscore_support,plot_confusion_matrix,precision_score,recall_score

### Plotting defaults 
plt.rc('text',usetex=True)
plt.rc('font',**{'family':'sans-serif','sans-serif':['Avant Garde']})

### Hyperparamters for files of the ANN model
rm_ensemble_mean = True

if rm_ensemble_mean == False:
    vari_predict = ['OHC100']
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
    vari_predict= ['OHC100']
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
dirname = '/Users/zlabe/Documents/Research/GmstTrendPrediction/SavedModels/'
savename = 'ANNv2_'+vari_predict[0]+'_hiatus_' + actFun + '_L2_'+ str(ridgePenalty)+ '_LR_' + str(lr_here)+ '_Batch'+ str(batch_size)+ '_Iters' + str(n_epochs) + '_' + str(len(hidden)) + 'x' + str(hidden[0]) + '_SegSeed' + str(random_segment_seed) + '_NetSeed'+ str(random_network_seed) 

if(rm_ensemble_mean==True):
    savename = savename + '_EnsembleMeanRemoved' 

### Directories to save files
directorydata = '/Users/zlabe/Documents/Research/GmstTrendPrediction/Data/'
directoryfigure = '/Users/zlabe/Desktop/GmstTrendPrediction/ANN_v2/Raw/'

###############################################################################
###############################################################################
###############################################################################
### Read in data for actual hiatuses in training
trainindices = np.genfromtxt(directorydata + 'trainingEnsIndices_' + savename + '.txt')
actual_train = np.genfromtxt(directorydata + 'trainingTrueLabels_' + savename + '.txt')
act_retrain = np.swapaxes(actual_train.reshape(trainindices.shape[0],1,yearsall.shape[0]),0,1).squeeze()

###############################################################################
###############################################################################
###############################################################################
### Read in data for actual hiatuses in testing
testindices = np.genfromtxt(directorydata + 'testingEnsIndices_' + savename + '.txt')
actual_test = np.genfromtxt(directorydata + 'testingTrueLabels_' + savename + '.txt')
act_retest = np.swapaxes(actual_test.reshape(testindices.shape[0],1,yearsall.shape[0]),0,1).squeeze()

###############################################################################
###############################################################################
###############################################################################
### Read in data for actual hiatuses in validation
valindices = np.genfromtxt(directorydata + 'validationEnsIndices_' + savename + '.txt')
actual_val = np.genfromtxt(directorydata + 'validationTrueLabels_' + savename + '.txt')
act_reval = np.swapaxes(actual_val.reshape(valindices.shape[0],1,yearsall.shape[0]),0,1).squeeze()

### Count number of hiatus per ensemble member
traincount = np.count_nonzero(act_retrain==1,axis=1)
testcount = np.count_nonzero(act_retest==1,axis=1)
valcount = np.count_nonzero(act_reval==1,axis=1)

### Concatenate
totalcounts = np.concatenate((traincount,testcount,valcount))

###############################################################################
###############################################################################
###############################################################################
### Create plot for histograms of slopes
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
        
fig = plt.figure()
ax = plt.subplot(111)
adjust_spines(ax, ['left','bottom'])            
ax.spines['top'].set_color('none')
ax.spines['right'].set_color('none') 
ax.spines['bottom'].set_color('dimgrey')
ax.spines['left'].set_color('dimgrey')
ax.spines['bottom'].set_linewidth(2)
ax.spines['left'].set_linewidth(2) 
ax.tick_params('both',length=5.5,width=2,which='major',color='dimgrey')  
ax.yaxis.grid(zorder=1,color='dimgrey',alpha=0.35)

### Plot histograms
weights_count = np.ones_like(totalcounts)/len(totalcounts)
n_count, bins_count, patches_count = plt.hist(totalcounts,bins=np.arange(0,21,1)-0.5,
                                        density=False,alpha=0.9,
                                        label=r'\textbf{HIATUS}',
                                        weights=weights_count,zorder=3)
for i in range(len(patches_count)):
    patches_count[i].set_facecolor('teal')
    patches_count[i].set_edgecolor('white')
    patches_count[i].set_linewidth(0.9)

plt.ylabel(r'\textbf{Proportion}',fontsize=10,color='k')
plt.xlabel(r'\textbf{Number of hiatus events per ensemble member}',fontsize=10,color='k')
plt.yticks(np.arange(0,1.1,0.1),map(str,np.round(np.arange(0,1.1,0.1),2)),size=6)
plt.xticks(np.arange(0,21,2),map(str,np.round(np.arange(0,21,2),2)),size=6)
plt.xlim([0,20])   
plt.ylim([0,0.3])
    
plt.savefig(directoryfigure + 'Histogram_HiatusCountEnsemble.png',
            dpi=300)