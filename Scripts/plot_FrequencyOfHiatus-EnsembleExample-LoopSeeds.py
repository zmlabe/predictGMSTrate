"""
Create plot to look at frequency of predictions in one ensemble member and 
compare that to the IPO

Author     : Zachary M. Labe
Date       : 4 October 2021
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
yearsall = np.arange(1990,2090+1,1)
directorymodel = '/Users/zlabe/Documents/Research/GmstTrendPrediction/SavedModels/'
directorydata = '/Users/zlabe/Documents/Research/GmstTrendPrediction/Data/'
directorydataIPO = '/Users/zlabe/Documents/Research/GmstTrendPrediction/Data/IPO/'
directorydataLOOP = '/Users/zlabe/Documents/Research/GmstTrendPrediction/Data/LoopSeeds/'
directoryfigure = '/Users/zlabe/Desktop/GmstTrendPrediction/ANN_v2/Testing/'

### Read in IPO index
IPO = np.genfromtxt(directorydataIPO + 'IPO_CESM2LE_1990-2099.txt',unpack=True)
IPOtest = IPO.transpose()[:,:yearsall.shape[0]]

### Read in seeds
seeds = np.load(directorydataLOOP + 'LoopSeedsResultsfor_ANNv2_OHC100_hiatus_EnsembleMeanRemoved_SEEDS.npz')
random_segment_seedq = seeds['random_segment_seedall']
random_network_seedq = seeds['random_network_seedall']

sizeOfTesting = 6
actual_test = np.empty((COUNTER,yearsall.shape[0]*sizeOfTesting))
predic_test = np.empty((COUNTER,yearsall.shape[0]*sizeOfTesting))
test_indices = np.empty((COUNTER,sizeOfTesting))
save_actex = []
save_preex = []
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
    else:
        print(ValueError('SOMETHING IS WRONG WITH DATA PROCESSING!'))
        sys.exit()
    
    ### Naming conventions for files
    savename = 'LoopSeedsResultsfor_ANNv2_'+vari_predict[0]+'_hiatus_' + actFun + '_L2_'+ str(ridgePenalty)+ '_LR_' + str(lr_here)+ '_Batch'+ str(batch_size)+ '_Iters' + str(n_epochs) + '_' + str(len(hidden)) + 'x' + str(hidden[0]) + '_SegSeed' + str(random_segment_seed) + '_NetSeed'+ str(random_network_seed)
    
    if(rm_ensemble_mean==True):
        savename = savename + '_EnsembleMeanRemoved'  
        
    pred = np.load(directorydataLOOP + savename + '_PREDICTIONS_%s.npz' % lo)
    actual_test[lo,:] = pred['actual_classtest']
    predic_test[lo,:] = pred['ypred_picktest']
    test_indices[lo,:] = pred['testIndices']
    
    ### Sample one ensemble member example
    sampleENS = 19
    actual_testq = pred['actual_classtest']
    predic_testq = pred['ypred_picktest']
    test_indicesq = pred['testIndices']
    act_req = np.swapaxes(actual_testq.reshape(test_indices.shape[1],1,yearsall.shape[0]),1,2).squeeze()
    pre_req = np.swapaxes(predic_testq.reshape(test_indices.shape[1],1,yearsall.shape[0]),1,2).squeeze()
    
    locens = np.where((test_indicesq == sampleENS))[0]
    if locens.size:
        save_actex.append(act_req[locens,:])
        save_preex.append(pre_req[locens,:])
    elif locens.size > 1:
        print(ValueError('SOMETHING IS WRONG WITH CODE - ENSEMBLE MEMBERS'))
        sys.exit()
    
### Reshape arrays for [ensemble,year]
act_re = np.swapaxes(actual_test.reshape(COUNTER,test_indices.shape[1],1,yearsall.shape[0]),1,2).squeeze()
pre_re = np.swapaxes(predic_test.reshape(COUNTER,test_indices.shape[1],1,yearsall.shape[0]),1,2).squeeze()

### Create arrays for frequency
save_actex = np.asarray(save_actex).squeeze()
save_preex = np.asarray(save_preex).squeeze()
print('\nSIZE OF FREQUENCY == %s!\n' % save_actex.shape[0])

### Calculate frequency of predictions
countens = np.count_nonzero(save_preex==1,axis=0)
freq_ens = countens/save_actex.shape[0]

### Pick right IPO member
IPOens = IPOtest[sampleENS,:]

###############################################################################
###############################################################################
###############################################################################
### Begin plot
fig = plt.figure(figsize=(8,6))
ax = plt.subplot(211)

adjust_spines(ax, ['left', 'bottom'])
ax.spines['top'].set_color('none')
ax.spines['right'].set_color('none')
ax.spines['left'].set_color('dimgrey')
ax.spines['bottom'].set_color('dimgrey')
ax.spines['left'].set_linewidth(2)
ax.spines['bottom'].set_linewidth(2)
ax.tick_params('both',length=4,width=2,which='major',color='dimgrey')
ax.yaxis.grid(zorder=1,color='dimgrey',alpha=0.35,clip_on=False)

rects = plt.bar(yearsall,save_actex[0,:])
plt.plot(yearsall,freq_ens,linewidth=4,color='maroon',alpha=2,zorder=3,clip_on=False,
          linestyle='--',dashes=(1,0.3))
for i in range(len(rects)):
    rects[i].set_color('maroon')
    rects[i].set_edgecolor('w')
    rects[i].set_alpha(0.4)

plt.text(1990,1.05,r'\textbf{[a]}',color='k',fontsize=7,ha='center')  
plt.yticks(np.arange(0,2,0.1),map(str,np.round(np.arange(0,2,0.1),2)),size=6)
plt.xticks(np.arange(1990,2100+1,10),map(str,np.arange(1990,2100+1,10)),size=6)
plt.xlim([1990,2090])   
plt.ylim([0,1])  

plt.text(2090,0.9,r'\textbf{ACTUAL SLOWDOWN}',fontsize=17,color='maroon',alpha=0.4,ha='right')
plt.text(2090,0.8,r'\textbf{{PREDICTED SLOWDOWN}',fontsize=17,color='maroon',alpha=1,ha='right') 
plt.ylabel(r'\textbf{Frequency of Classification}',color='k',fontsize=10)    

###############################################################################
ax = plt.subplot(212)

### Begin plot
adjust_spines(ax, ['left', 'bottom'])
ax.spines['top'].set_color('none')
ax.spines['right'].set_color('none')
ax.spines['left'].set_color('dimgrey')
ax.spines['bottom'].set_color('dimgrey')
ax.spines['left'].set_linewidth(2)
ax.spines['bottom'].set_linewidth(2)
ax.tick_params('both',length=4,width=2,which='major',color='dimgrey')
ax.yaxis.grid(zorder=1,color='dimgrey',alpha=0.35,clip_on=False)

IPOens_masked = np.ma.masked_less_equal(IPOens, 0)

plt.bar(yearsall,IPOens,color='deepskyblue',
        edgecolor='darkblue',zorder=9,linewidth=0.3) 
plt.bar(yearsall,IPOens_masked,
        color='crimson',edgecolor='darkred',zorder=9,clip_on=False,
        linewidth=0.3) 

plt.yticks(np.arange(-5,5,0.5),map(str,np.round(np.arange(-5,6,0.5),2)),size=6)
plt.xticks(np.arange(1990,2100+1,10),map(str,np.arange(1990,2100+1,10)),size=6)
plt.xlim([1990,2090])    
plt.ylim([-2.5,2.5])  

plt.text(1990,2.77,r'\textbf{[b]}',color='k',fontsize=7,ha='center')  
plt.ylabel(r'\textbf{Unfiltered IPO Index}',color='k',fontsize=10)    
        
plt.tight_layout()
plt.subplots_adjust(hspace=0.4)

if rm_ensemble_mean == True:
    plt.savefig(directoryfigure + 'FrequencyOfHiatus_LoopSeeds_EnsembleExample_rmENSEMBLEmean_indexens-%s.png' % sampleENS,dpi=300)
else:
    plt.savefig(directoryfigure + 'FrequencyOfHiatus_EnsembleExample_LoopSeeds_indexens-%s.png' % sampleENS,dpi=300)