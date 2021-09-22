"""
Create plots to compare accuracy scores for different class weights in the 
same ANN architecture for validation data only

Author     : Zachary M. Labe
Date       : 21 September 2021
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
from sklearn.metrics import accuracy_score,precision_recall_fscore_support,precision_score,recall_score,f1_score

### Plotting defaults 
plt.rc('text',usetex=True)
plt.rc('font',**{'family':'sans-serif','sans-serif':['Avant Garde']})

### Hyperparamters for files of the ANN model
rm_ensemble_mean = True
lengthOfWeightTests = 11
directoryfigure = '/Users/zlabe/Desktop/GmstTrendPrediction/ANN_v2/Scores/'

acc_trainl = np.zeros((lengthOfWeightTests))
acc_testl = np.zeros((lengthOfWeightTests))
acc_vall = np.zeros((lengthOfWeightTests))

prec_trainl = np.zeros((lengthOfWeightTests))
prec_testl = np.zeros((lengthOfWeightTests))
prec_vall = np.zeros((lengthOfWeightTests))

recall_trainl = np.zeros((lengthOfWeightTests))
recall_testl = np.zeros((lengthOfWeightTests))
recall_vall = np.zeros((lengthOfWeightTests))

f1_trainl = np.zeros((lengthOfWeightTests))
f1_testl = np.zeros((lengthOfWeightTests))
f1_vall = np.zeros((lengthOfWeightTests))
for lo in range(lengthOfWeightTests):
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
        fractWeight = np.arange(0.1,1.2,0.1)
        yearsall = np.arange(1990,2099+1,1)
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
        fractWeight = np.arange(0.1,1.2,0.1)
        yearsall = np.arange(1990,2099+1,1)
    else:
        print(ValueError('SOMETHING IS WRONG WITH DATA PROCESSING!'))
        sys.exit()
    
    ### Naming conventions for files
    dirname = '/Users/zlabe/Documents/Research/GmstTrendPrediction/SavedModels/'
    directorydata = '/Users/zlabe/Documents/Research/GmstTrendPrediction/Data/LoopWeights/scores/'
    savename = 'ANNv2_'+vari_predict[0]+'_hiatus_' + actFun + '_L2_'+ str(ridgePenalty)+ '_LR_' + str(lr_here)+ '_Batch'+ str(batch_size)+ '_Iters' + str(n_epochs) + '_' + str(len(hidden)) + 'x' + str(hidden[0]) + '_SegSeed' + str(random_segment_seed) + '_NetSeed'+ str(random_network_seed) + '_fracWeights' + str(np.round(fractWeight[lo],2)) + '_%s' % lo 
    
    if(rm_ensemble_mean==True):
        savename = savename + '_EnsembleMeanRemoved'  
    
    ### Directories to save files
    scoresall = np.load(directorydata + 'metrics_LoopWeights_%s.npz' % (savename))
    acc_trainl[lo] = scoresall['acctrain']*100
    acc_testl[lo] = scoresall['acctest']*100
    acc_vall[lo] = scoresall['accval']*100
    
    prec_trainl[lo] = scoresall['prectrain']*100
    prec_testl[lo] = scoresall['prectest']*100
    prec_vall[lo] = scoresall['precval']*100
    
    recall_trainl[lo] = scoresall['recalltrain']*100
    recall_testl[lo] = scoresall['recalltest']*100
    recall_vall[lo] = scoresall['recallval']*100
    
    f1_trainl[lo] = scoresall['f1_train']*100
    f1_testl[lo] = scoresall['f1_test']*100
    f1_vall[lo] = scoresall['f1_val']*100
    
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

fig = plt.figure(figsize=(10,4))
ax = plt.subplot(141)
adjust_spines(ax, ['left', 'bottom'])
ax.spines['top'].set_color('none')
ax.spines['right'].set_color('none')
ax.spines['left'].set_color('dimgrey')
ax.spines['bottom'].set_color('dimgrey')
ax.spines['left'].set_linewidth(2)
ax.spines['bottom'].set_linewidth(2)
ax.tick_params('both',length=4,width=2,which='major',color='dimgrey')
ax.yaxis.grid(zorder=1,color='dimgrey',alpha=0.2,clip_on=False)

plt.plot(fractWeight,acc_vall,color='maroon',linewidth=4,linestyle='-',
         clip_on=False)
    
plt.xticks(np.arange(0.1,1.2,0.1),map(str,np.round(np.arange(0.1,1.2,0.1),2)),size=6)
plt.yticks(np.arange(0,101,10),map(str,np.arange(0,101,10)),size=6)
plt.xlim([0.1,1.1])   
plt.ylim([0,100])
plt.ylabel(r'\textbf{Score \%}')
plt.text(1.11,2,r'\textbf{ACCURACY}',fontsize=18,color='dimgrey',
         ha='right')

###############################################################################
###############################################################################
###############################################################################
ax = plt.subplot(142)
adjust_spines(ax, ['left', 'bottom'])
ax.spines['top'].set_color('none')
ax.spines['right'].set_color('none')
ax.spines['left'].set_color('dimgrey')
ax.spines['bottom'].set_color('dimgrey')
ax.spines['left'].set_linewidth(2)
ax.spines['bottom'].set_linewidth(2)
ax.tick_params('both',length=4,width=2,which='major',color='dimgrey')
ax.yaxis.grid(zorder=1,color='dimgrey',alpha=0.2,clip_on=False)

plt.plot(fractWeight,prec_vall,color='maroon',linewidth=4,linestyle='-',
         clip_on=False,label=r'\textbf{VALIDATION}')
    
plt.xticks(np.arange(0.1,1.2,0.1),map(str,np.round(np.arange(0.1,1.2,0.1),2)),size=6)
plt.yticks(np.arange(0,101,10),map(str,np.arange(0,101,10)),size=6)
plt.xlim([0.1,1.1])   
plt.ylim([0,100])
plt.text(0.73,-13,r'\textbf{Fraction of Class Weights on Hiatus}',fontsize=7,color='k')
plt.text(1.11,2,r'\textbf{PRECISION}',fontsize=18,color='dimgrey',
         ha='right')

leg = plt.legend(shadow=False,fontsize=11,loc='upper center',
              bbox_to_anchor=(1.08,1.14),fancybox=True,ncol=4,frameon=False,
              handlelength=5,handletextpad=1)
for line,text in zip(leg.get_lines(), leg.get_texts()):
    text.set_color(line.get_color())

###############################################################################
###############################################################################
###############################################################################
ax = plt.subplot(143)
adjust_spines(ax, ['left', 'bottom'])
ax.spines['top'].set_color('none')
ax.spines['right'].set_color('none')
ax.spines['left'].set_color('dimgrey')
ax.spines['bottom'].set_color('dimgrey')
ax.spines['left'].set_linewidth(2)
ax.spines['bottom'].set_linewidth(2)
ax.tick_params('both',length=4,width=2,which='major',color='dimgrey')
ax.yaxis.grid(zorder=1,color='dimgrey',alpha=0.2,clip_on=False)

plt.plot(fractWeight,recall_vall,color='maroon',linewidth=4,linestyle='-',
         clip_on=False)
    
plt.xticks(np.arange(0.1,1.2,0.1),map(str,np.round(np.arange(0.1,1.2,0.1),2)),size=6)
plt.yticks(np.arange(0,101,10),map(str,np.arange(0,101,10)),size=6)
plt.xlim([0.1,1.1])   
plt.ylim([0,100])
plt.text(1.11,2,r'\textbf{RECALL}',fontsize=18,color='dimgrey',
         ha='right')

###############################################################################
###############################################################################
###############################################################################
ax = plt.subplot(144)
adjust_spines(ax, ['left', 'bottom'])
ax.spines['top'].set_color('none')
ax.spines['right'].set_color('none')
ax.spines['left'].set_color('dimgrey')
ax.spines['bottom'].set_color('dimgrey')
ax.spines['left'].set_linewidth(2)
ax.spines['bottom'].set_linewidth(2)
ax.tick_params('both',length=4,width=2,which='major',color='dimgrey')
ax.yaxis.grid(zorder=1,color='dimgrey',alpha=0.2,clip_on=False)

plt.plot(fractWeight,f1_vall,color='maroon',linewidth=4,linestyle='-',
         clip_on=False)
    
plt.xticks(np.arange(0.1,1.2,0.1),map(str,np.round(np.arange(0.1,1.2,0.1),2)),size=6)
plt.yticks(np.arange(0,101,10),map(str,np.arange(0,101,10)),size=6)
plt.xlim([0.1,1.1])   
plt.ylim([0,100])
plt.text(1.11,2,r'\textbf{F1-SCORE}',fontsize=18,color='dimgrey',
         ha='right')

if rm_ensemble_mean == True:
    plt.savefig(directoryfigure + 'ValidationScores-LoopWeights_Hiatus_EDA-v2_rmENSEMBLEmean.png',dpi=300)
else:
    plt.savefig(directoryfigure + 'ValidationScores-LoopWeights_Hiatus_EDA-v2.png',dpi=300)
    
    