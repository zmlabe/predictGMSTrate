"""
Plots confidence comparisons for right/wrong hiatus

Author     : Zachary M. Labe
Date       : 30 September 2021
Version    : 2 (mostly for testing)
"""

### Import packages
import sys
import matplotlib.pyplot as plt
import numpy as np
import palettable.wesanderson as ww

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
directoryfigure = '/Users/zlabe/Desktop/GmstTrendPrediction/ANN_v2/Testing/'

###############################################################################
###############################################################################
###############################################################################
### Read in data for testing predictions and actual hiatuses
testindices = np.genfromtxt(directorydata + 'testingEnsIndices_' + savename + '.txt')
actual_test = np.genfromtxt(directorydata + 'testingTrueLabels_' + savename + '.txt')
predict_test = np.genfromtxt(directorydata + 'testingPredictedLabels_' + savename+ '.txt')
predict_testconf = np.genfromtxt(directorydata + 'testingPredictedConfidence_' + savename+ '.txt')
predict_testconfh = predict_testconf[:,0]
predict_testconfc = predict_testconf[:,1]

act_retest = np.swapaxes(actual_test.reshape(testindices.shape[0],1,yearsall.shape[0]),0,1).squeeze()
predict_retest = np.swapaxes(predict_test.reshape(testindices.shape[0],1,yearsall.shape[0]),0,1).squeeze()
predict_retestconfh = np.swapaxes(predict_testconfh.reshape(testindices.shape[0],1,yearsall.shape[0]),0,1).squeeze()
predict_retestconfc = np.swapaxes(predict_testconfc.reshape(testindices.shape[0],1,yearsall.shape[0]),0,1).squeeze()

conf_hh = []
conf_ch = []
conf_hc = []
conf_cc = []
for i in range(testindices.shape[0]):
    for yr in range(yearsall.shape[0]):
        if (act_retest[i,yr] == 1) and (predict_retest[i,yr] == 1):
            conf_hh.append(predict_retestconfc[i,yr])
        if (act_retest[i,yr] == 0) and (predict_retest[i,yr] == 1):
            conf_ch.append(predict_retestconfc[i,yr])
        if (act_retest[i,yr] == 1) and (predict_retest[i,yr] == 0):
            conf_hc.append(predict_retestconfh[i,yr])
        if (act_retest[i,yr] == 0) and (predict_retest[i,yr] == 0):
            conf_cc.append(predict_retestconfh[i,yr])
            
mean_hh = np.nanmean(conf_hh)
mean_ch = np.nanmean(conf_ch)
mean_hc = np.nanmean(conf_hc)
mean_cc = np.nanmean(conf_cc)
conf95_hh = np.percentile(conf_hh,95)
conf95_ch = np.percentile(conf_ch,95)
conf95_hc = np.percentile(conf_hc,95)
conf95_cc = np.percentile(conf_cc,95)
conf5_hh = np.percentile(conf_hh,5)
conf5_ch = np.percentile(conf_ch,5)
conf5_hc = np.percentile(conf_hc,5)
conf5_cc = np.percentile(conf_cc,5)
            
allconf = [conf_hh,conf_ch,conf_hc,conf_cc]
meanconf = [mean_hh,mean_ch,mean_hc,mean_cc]
conf95 = [conf95_hh,conf95_ch,conf95_hc,conf95_cc]
conf5 = [conf5_hh,conf5_ch,conf5_hc,conf5_cc]
            
###############################################################################
###############################################################################
###############################################################################    
### Create graph 
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
### Training figure
fig = plt.figure()
ax = plt.subplot(111)
adjust_spines(ax, ['left', 'bottom'])
ax.spines['top'].set_color('none')
ax.spines['right'].set_color('none')
ax.spines['left'].set_color('dimgrey')
ax.spines['bottom'].set_color('none')
ax.spines['left'].set_linewidth(2)
ax.tick_params('both',length=4,width=2,which='major',color='dimgrey',
               labelbottom='off',bottom='off')
ax.tick_params(axis = "x", which = "both", bottom = False, top = False)
ax.yaxis.grid(zorder=1,color='dimgrey',alpha=0.35,clip_on=False)

labelx = [r'\textbf{[H$_{act}$,H$_{pre}$]}',
          r'\textbf{[CC$_{act}$,H$_{pre}$]}',
          r'\textbf{[H$_{act}$,CC$_{pre}$]}',
          r'\textbf{[CC$_{act}$,CC$_{pre}$]}']

ccc=ww.FantasticFox2_5.mpl_colormap(np.linspace(0.3,1,len(allconf)))
for i in range(len(allconf)):
    plt.scatter(i,meanconf[i],s=150,c=ccc[i],edgecolor=ccc[i],zorder=5,clip_on=False)
    plt.errorbar(i,meanconf[i],
                  yerr=np.array([[meanconf[i]-conf5[i],conf95[i]-meanconf[i]]]).T,
                  color=ccc[i],linewidth=1.5,capthick=4,capsize=10,clip_on=False)

plt.ylabel(r'\textbf{Confidence}',color='k',fontsize=11)    
plt.xticks(np.arange(0,4,1),labelx)
plt.yticks(np.arange(0,1.1,0.2),map(str,np.round(np.arange(0,1.1,0.2),2)),size=6)
plt.xlim([-1,4])
plt.ylim([0,1.0])
plt.savefig(directoryfigure + 'TestingConfidence_errorbar.png',dpi=300)