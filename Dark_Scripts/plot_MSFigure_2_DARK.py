"""
Make figure of predictions of testing data for paper

Author     : Zachary M. Labe
Date       : 6 October 2021
Version    : 2 
"""

### Import packages
import sys
import matplotlib.pyplot as plt
import numpy as np

### Plotting defaults 
plt.rc('text',usetex=True)
plt.rc('font',**{'family':'sans-serif','sans-serif':['Avant Garde']}) 
plt.rc('savefig',facecolor='black')
plt.rc('axes',edgecolor='darkgrey')
plt.rc('xtick',color='w')
plt.rc('ytick',color='w')
plt.rc('axes',labelcolor='darkgrey')
plt.rc('axes',facecolor='black')

### Hyperparamters for files of the ANN model
rm_ensemble_mean = True
ens1 = np.arange(1,10+1,1)
ens2 = np.arange(21,50+1,1)
ens = np.append(ens1,ens2)

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
    vari_predict = ['OHC100']
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
directoryfigure = '/Users/zlabe/Documents/Projects/predictGMSTrate/Dark_Figures/'

###############################################################################
###############################################################################
###############################################################################
### Read in data for testing predictions and actual hiatuses
testindices = np.genfromtxt(directorydata + 'testingEnsIndices_' + savename + '.txt').astype(int)
actual_test = np.genfromtxt(directorydata + 'testingTrueLabels_' + savename + '.txt')
predict_test = np.genfromtxt(directorydata + 'testingPredictedLabels_' + savename+ '.txt')
predict_testconf = np.genfromtxt(directorydata + 'testingPredictedConfidence_' + savename+ '.txt')[:,1]

### Reshape arrays for [ensemble,year]
act_re = np.swapaxes(actual_test.reshape(testindices.shape[0],1,yearsall.shape[0]),0,1).squeeze()
pre_re = np.swapaxes(predict_test.reshape(testindices.shape[0],1,yearsall.shape[0]),0,1).squeeze()
pre_reconf = np.swapaxes(predict_testconf.reshape(testindices.shape[0],1,yearsall.shape[0]),0,1).squeeze()

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
            spine.set_position(('outward',10))
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

fig = plt.figure(figsize=(11,4))
ax = plt.subplot(111)
adjust_spines(ax, ['left', 'bottom'])
ax.spines['top'].set_color('none')
ax.spines['right'].set_color('none')
ax.spines['left'].set_color('none')
ax.spines['bottom'].set_color('darkgrey')
ax.spines['left'].set_linewidth(2)
ax.spines['bottom'].set_linewidth(2)
ax.tick_params('both',length=4,width=2,which='major',color='darkgrey')
ax.tick_params(axis='y',which='both',length=0)
ax.yaxis.grid(zorder=1,color='darkgrey',alpha=0.8,clip_on=False)

for i in range(testindices.shape[0]):
    plt.scatter(yearsall,act_re[i],s=110,color='w',clip_on=False,zorder=2,
                edgecolor='k',linewidth=0.1,label='Actual Slowdowns')
    for yr in range(pre_re.shape[1]):
        if act_re[i,yr] == pre_re[i,yr]:
            cc = 'deepskyblue'
            label = 'Correct Predictions'
        elif act_re[i,yr] != pre_re[i,yr]:
            cc = 'r'
            label = 'Wrong Predictions'
        else:
            print(ValueError('SOMETHING MIGHT BE WRONG!'))
            sys.exit()
        plt.scatter(yearsall[yr],pre_re[i,yr],s=45,color=cc,clip_on=False,
                    zorder=3,edgecolor='darkgrey',linewidth=0.1,label=label)
        if i == 0:
            if yr == 3:
                leg = plt.legend(shadow=False,fontsize=10,loc='upper center',
                                 bbox_to_anchor=(0.5,1.175),
                                 fancybox=True,ncol=3,frameon=False,
                                 handlelength=0)
                for text in leg.get_texts():
                    text.set_color('w')
    
plt.xticks(np.arange(1990,2101,10),map(str,np.arange(1990,2101,10)),size=7)
plt.yticks(np.arange(1,testindices.shape[0]+1,1),map(str,ens[testindices]),size=10)
plt.xlim([1990,2090])   
plt.ylim([1,testindices.shape[0]])
plt.ylabel(r'\textbf{Ensemble Member \#}',color='w')

plt.savefig(directoryfigure + 'Figure_2_DARK.png',dpi=600)


