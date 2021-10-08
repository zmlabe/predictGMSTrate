"""
Create manuscript figure for observations

Author     : Zachary M. Labe
Date       : 6 October 2021
Version    : 2 
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
directoryfigure = '/Users/zlabe/Desktop/GmstTrendPrediction/MS-Figures_v1/'

### Read in seeds
seeds = np.load(directorydata + 'LoopSeedsResultsfor_ANNv2_OHC100_hiatus_EnsembleMeanRemoved_SEEDS.npz')
random_segment_seedq = seeds['random_segment_seedall']
random_network_seedq = seeds['random_network_seedall']

### Obs parameters
years = np.arange(1990,2020+1,1)

actual_obs = np.empty((COUNTER,years.shape[0]))
pred_obs = np.empty((COUNTER,years.shape[0]))
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
    
    pred = np.load(directorydata + savename + '_PREDICTIONS_%s.npz' % lo)
    actual_obs[lo,:] = pred['actualobs']
    pred_obs[lo,:] = pred['selectobs']
    
### Calculate frequency of predictions
countobs = np.count_nonzero(pred_obs==1,axis=0)
freq_obs = countobs/COUNTER

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

rects = plt.bar(years,actual_obs[0,:])
plt.plot(years,freq_obs,linewidth=4,color='maroon',alpha=2,zorder=3,clip_on=False,
          linestyle='--',dashes=(1,0.3))
for i in range(len(rects)):
    rects[i].set_color('teal')
    rects[i].set_edgecolor('w')
    rects[i].set_alpha(0.4)
    
plt.fill_between(x=years[-10:],y1=0,y2=1,facecolor='darkgrey',zorder=0,
             alpha=0.3,edgecolor='none')

plt.text(1990,1.05,r'\textbf{[a]}',color='k',fontsize=7,ha='center')  
plt.yticks(np.arange(0,2,0.1),map(str,np.round(np.arange(0,2,0.1),2)),size=6)
plt.xticks(np.arange(1990,2030+1,5),map(str,np.arange(1990,2030+1,5)),size=6)
plt.xlim([1990,2020])   
plt.ylim([0,1])  

plt.text(1990,0.9,r'\textbf{ACTUAL SLOWDOWN}',fontsize=16,color='teal',alpha=0.4)
plt.text(1990,0.8,r'\textbf{{PREDICTED SLOWDOWN}',fontsize=16,color='maroon',alpha=1) 
plt.ylabel(r'\textbf{Frequency of Classification}',color='k',fontsize=10)    

###############################################################################
ax = plt.subplot(212)

### Read in IPO index for observations
directoryoutput = '/Users/zlabe/Documents/Research/GmstTrendPrediction/Data/IPO/'
IPO = np.genfromtxt(directoryoutput + 'IPO_ERA5_1990-2020.txt')

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

IPO_masked = np.ma.masked_less_equal(IPO, 0)

plt.bar(years,IPO,color='deepskyblue',
        edgecolor='darkblue',zorder=9,linewidth=0.3) 
plt.bar(years,IPO_masked,
        color='crimson',edgecolor='darkred',zorder=9,clip_on=False,
        linewidth=0.3) 

rects = plt.bar(years,actual_obs[0,:]*2.5)
for i in range(len(rects)):
    rects[i].set_color('teal')
    rects[i].set_edgecolor('w')
    rects[i].set_alpha(0.4)
rects = plt.bar(years,actual_obs[0,:]*-2.5)
for i in range(len(rects)):
    rects[i].set_color('teal')
    rects[i].set_edgecolor('w')
    rects[i].set_alpha(0.4)

plt.yticks(np.arange(-5,5,0.5),map(str,np.round(np.arange(-5,6,0.5),2)),size=6)
plt.xticks(np.arange(1990,2030+1,5),map(str,np.arange(1990,2030+1,5)),size=6)
plt.xlim([1990,2020])   
plt.ylim([-2.5,2.5])  

plt.text(1990,2.77,r'\textbf{[b]}',color='k',fontsize=7,ha='center')  
plt.ylabel(r'\textbf{Unfiltered IPO Index}',color='k',fontsize=10)    
        
plt.tight_layout()
plt.subplots_adjust(hspace=0.4)

plt.savefig(directoryfigure + 'Figure_4.png',dpi=600)