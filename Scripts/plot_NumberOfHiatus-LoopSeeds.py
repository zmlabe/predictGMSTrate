"""
Create plot to show scores vs. number of hiatuses it trained on

Author     : Zachary M. Labe
Date       : 27 September 2021
Version    : 2 (mostly for testing)
"""

### Import packages
import sys
import matplotlib.pyplot as plt
import matplotlib.colors as c
import numpy as np
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

### Hyperparamters for files of the ANN model
rm_ensemble_mean = True
COUNTER = 100
directorydata = '/Users/zlabe/Documents/Research/GmstTrendPrediction/Data/LoopSeeds/'
directoryfigure = '/Users/zlabe/Desktop/GmstTrendPrediction/ANN_v2/Scores/'

### Read in seeds
seeds = np.load(directorydata + 'LoopSeedsResultsfor_ANNv2_OHC100_hiatus_EnsembleMeanRemoved_SEEDS.npz')
random_segment_seedq = seeds['random_segment_seedall']
random_network_seedq = seeds['random_network_seedall']

acctest = np.empty((COUNTER))
prectest = np.empty((COUNTER))
recalltest = np.empty((COUNTER))
f1test = np.empty((COUNTER))
act_train = np.empty((COUNTER,28*101))
for lo in range(COUNTER):
    if rm_ensemble_mean == True:
        vari_predict = ['OHC100']
        fac = 0.7
        random_segment_seed = random_segment_seedq[lo]
        random_network_seed = random_network_seedq[lo]
        hidden = [30,30]
        n_epochs = 500
        batch_size = 128
        lr_here = 0.001
        ridgePenalty = 0.5
        actFun = 'relu'
        fractWeight = np.arange(0.1,1.2,0.1)
        yearsall = np.arange(1990,2090+1,1)
    else:
        print(ValueError('SOMETHING IS WRONG WITH DATA PROCESSING!'))
        sys.exit()
    
    ### Naming conventions for files
    savename = 'LoopSeedsResultsfor_ANNv2_'+vari_predict[0]+'_hiatus_' + actFun + '_L2_'+ str(ridgePenalty)+ '_LR_' + str(lr_here)+ '_Batch'+ str(batch_size)+ '_Iters' + str(n_epochs) + '_' + str(len(hidden)) + 'x' + str(hidden[0]) + '_SegSeed' + str(random_segment_seed) + '_NetSeed'+ str(random_network_seed)
    
    if(rm_ensemble_mean==True):
        savename = savename + '_EnsembleMeanRemoved'  
        
    scores = np.load(directorydata + savename + '_SCORES_%s.npz' % lo)
    acctest[lo] = scores['acctest']
    prectest[lo] = scores['prectest']
    recalltest[lo] = scores['recalltest']
    f1test[lo] = scores['f1_test']
    
    pred = np.load(directorydata + savename + '_PREDICTIONS_%s.npz' % lo)
    act_train[lo,:] = pred['actual_classtrain']
    
### Gather data and place percent
alldata = np.asarray([acctest,prectest,recalltest,f1test]) * 100
class_train_re = np.swapaxes(act_train.reshape(COUNTER,28,1,101),1,2).squeeze()

act_re_seeds = class_train_re.reshape(COUNTER,28*101)
count_seeds = np.empty((COUNTER))
for i in range(COUNTER):
    count_seeds[i] = np.count_nonzero(class_train_re[i] == 1)
    
### Correlation
r,p = sts.pearsonr(count_seeds,f1test*100)
    
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
ax.yaxis.grid(zorder=1,color='dimgrey',alpha=0.35,clip_on=False)

plt.scatter(count_seeds,f1test*100,s=36,color='k',clip_on=False,
            zorder=3,edgecolor='dimgrey',linewidth=0.5)
    
plt.xticks(np.arange(260,320,10),map(str,np.arange(260,320,10)),size=6)
plt.yticks(np.arange(0,101,10),map(str,np.round(np.arange(0,101,10),2)),size=6)
plt.xlim([260,310])   
plt.ylim([0,50])
plt.ylabel(r'\textbf{F1-Score [\%] in Testing Data}',color='k',
           fontsize=10)
plt.xlabel(r'\textbf{Number of Hiatus Events in Training Data}',color='k',
           fontsize=10)

plt.text(310.2,1,r'\textbf{R$^{2}$=%s}' % (np.round(r**2,2)),
         color='k',fontsize=12,ha='right')

if rm_ensemble_mean == True:
    plt.savefig(directoryfigure + 'Scores-NumberOfHiatus_Hiatus_EDA-v2_rmENSEMBLEmean.png',dpi=300)
else:
    plt.savefig(directoryfigure + 'Scores-NumberOfHiatus_Hiatus_EDA-v2.png',dpi=300)