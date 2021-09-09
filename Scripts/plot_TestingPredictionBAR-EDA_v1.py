"""
First exploratory data analysis for making plots of testing accuracy using
bar graph for each ensemble member

Author     : Zachary M. Labe
Date       : 9 September 2021
Version    : 1 (mostly for testing)
"""

### Import packages
import sys
import matplotlib.pyplot as plt
import numpy as np

### Plotting defaults 
plt.rc('text',usetex=True)
plt.rc('font',**{'family':'sans-serif','sans-serif':['Avant Garde']})

### Hyperparamters for files of the ANN model
rm_ensemble_mean = False

if rm_ensemble_mean == False:
    variq = 'T2M'
    fac = 0.8
    random_segment_seed = int(np.genfromtxt('/Users/zlabe/Documents/Research/GmstTrendPrediction/Data/SelectedSegmentSeed.txt',unpack=True))
    random_network_seed = 87750
    hidden = [20,20]
    n_epochs = 500
    batch_size = 128
    lr_here = 0.001
    ridgePenalty = 0.05
    actFun = 'relu'
    fractWeight = 0.5
    yearsall = np.arange(1990,2099+1,1)
elif rm_ensemble_mean == True:
    variq = 'T2M'
    fac = 0.8
    random_segment_seed = int(np.genfromtxt('/Users/zlabe/Documents/Research/GmstTrendPrediction/Data/SelectedSegmentSeed.txt',unpack=True))
    random_network_seed = 87750
    hidden = [30,30]
    n_epochs = 500
    batch_size = 128
    lr_here = 0.001
    ridgePenalty = 0.35
    actFun = 'relu'
    fractWeight = 0.5
    yearsall = np.arange(1990,2099+1,1)
else:
    print(ValueError('SOMETHING IS WRONG WITH DATA PROCESSING!'))
    sys.exit()

### Naming conventions for files
directorymodel = '/Users/zlabe/Documents/Research/GmstTrendPrediction/SavedModels/'
savename = 'ANN_'+variq+'_hiatus_' + actFun + '_L2_'+ str(ridgePenalty)+ '_LR_' + str(lr_here)+ '_Batch'+ str(batch_size)+ '_Iters' + str(n_epochs) + '_' + str(len(hidden)) + 'x' + str(hidden[0]) + '_SegSeed' + str(random_segment_seed) + '_NetSeed'+ str(random_network_seed)
if(rm_ensemble_mean==True):
    savename = savename + '_EnsembleMeanRemoved'  
    
### Directories to save files
directorydata = '/Users/zlabe/Documents/Research/GmstTrendPrediction/Data/'
directoryfigure = '/Users/zlabe/Desktop/GmstTrendPrediction/ANN_v1/Testing/'

###############################################################################
###############################################################################
###############################################################################
### Read in data for testing predictions and actual hiatuses
testindices = np.genfromtxt(directorydata + 'testingEnsIndices_' + savename + '.txt')
actual_test = np.genfromtxt(directorydata + 'testingTrueLabels_' + savename + '.txt')
predict_test = np.genfromtxt(directorydata + 'testingPredictedLabels_' + savename+ '.txt')
predict_testconf = np.genfromtxt(directorydata + 'testingPredictedConfidence_' + savename+ '.txt')[:,1]

### Reshape arrays for [ensemble,year]
act_re = np.swapaxes(actual_test.reshape(testindices.shape[0],1,yearsall.shape[0]),0,1).squeeze()
pre_re = np.swapaxes(predict_test.reshape(testindices.shape[0],1,yearsall.shape[0]),0,1).squeeze()
pre_reconf = np.swapaxes(predict_testconf.reshape(testindices.shape[0],1,yearsall.shape[0]),0,1).squeeze()

### 10th percentile
conf10 = np.percentile(pre_reconf.ravel(),90)

###############################################################################
###############################################################################
###############################################################################
### Count hiatuses in testing
uniquetest,counttest = np.unique(predict_test,return_counts=True)
actual_uniquetest,actual_counttest = np.unique(actual_test,return_counts=True)

###############################################################################
###############################################################################
###############################################################################
### Create arrays for plotting
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

length = np.arange(yearsall.shape[0])

for i in range(testindices.shape[0]):
    act_re[i,np.where(act_re[i] == 1)] = i+1
    pre_re[i,np.where(pre_re[i] == 1)] = i+1
act_re[np.where(act_re == 0)] = np.nan
pre_re[np.where(pre_re == 0)] = np.nan

fig = plt.figure(figsize=(8,5))
for r in range(len(testindices)):
    ax = plt.subplot(2,len(testindices)//2,r+1)
    adjust_spines(ax, ['left', 'bottom'])
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.spines['left'].set_color('dimgrey')
    ax.spines['bottom'].set_color('dimgrey')
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    ax.tick_params('both',length=4,width=2,which='major',color='dimgrey')
    
    rects = plt.bar(yearsall,act_re[r])
    plt.plot(yearsall,pre_reconf[r],linewidth=0.8,color='maroon',alpha=1,zorder=3,
              linestyle='--',dashes=(1,0.3))
    for i in range(len(rects)):
        rects[i].set_color('teal')
        rects[i].set_edgecolor('teal')
        rects[i].set_alpha(0.6)
    
    if any([r==0,r==4]):
        plt.yticks(np.arange(0,2,0.1),map(str,np.round(np.arange(0,2,0.1),2)),size=6)
        plt.xticks(np.arange(1990,2101,30),map(str,np.arange(1990,2101,30)),size=6)
        plt.xlim([1990,2090])   
        plt.ylim([0,1])  
    else:
        ax.axes.yaxis.set_ticklabels([])
        plt.xticks(np.arange(1990,2101,30),map(str,np.arange(1990,2101,30)),size=6)
        plt.xlim([1990,2090])   
        plt.ylim([0,1])  
    
    plt.text(2087,1.0,r'\textbf{[\#%s]}' % (r+1),ha='center',va='center',color='k',fontsize=6)
fig.suptitle(r'\textbf{TESTING DATA BY ENSEMBLE MEMBER}',color='dimgrey',fontsize=20)
     
if rm_ensemble_mean == True:
    plt.savefig(directoryfigure + 'TestingConfidenceBAR_Hiatus_EDA-v1_rmENSEMBLEmean.png',dpi=300)
else:
    plt.savefig(directoryfigure + 'TestingConfidenceBAR_Hiatus_EDA-v1.png',dpi=300)

