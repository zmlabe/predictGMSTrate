"""
First exploratory data analysis for making plots of observations

Author     : Zachary M. Labe
Date       : 16 September 2021
Version    : 2 (mostly for testing)
"""

### Import packages
import sys
import matplotlib.pyplot as plt
import numpy as np
from netCDF4 import Dataset
import scipy.stats as stats
import palettable.cubehelix as cm
import cmocean as cmocean
import calc_Utilities as UT
import calc_dataFunctions as df
import calc_Stats as dSS
from sklearn.metrics import accuracy_score

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
    yearsall = np.arange(1990,2020+1,1)
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
    yearsall = np.arange(1990,2020+1,1)
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
directoryfigure = '/Users/zlabe/Desktop/GmstTrendPrediction/ANN_v2/Obs/'

###############################################################################
###############################################################################
###############################################################################
### Read in data for observations
predict_obs = np.genfromtxt(directorydata + 'obsLabels_' + savename + '.txt')
actual_obs = np.genfromtxt(directorydata + 'obsActualLabels_' + savename + '.txt')
confidence = np.genfromtxt(directorydata + 'obsConfid_' + savename + '.txt')

###############################################################################
###############################################################################
###############################################################################
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
ax.yaxis.grid(zorder=1,color='darkgrey',alpha=0.35,clip_on=False,linewidth=0.4)

plt.plot(yearsall,confidence[:,0],linewidth=4,color='teal',alpha=1,zorder=3,clip_on=False,
         label=r'\textbf{Climate Change}')
plt.plot(yearsall,confidence[:,1],linewidth=2,color='maroon',alpha=1,zorder=3,clip_on=False,
         linestyle='--',dashes=(1,0.3),label=r'\textbf{Hiatus}')

plt.fill_between(x=yearsall[-10:],y1=0,y2=1,facecolor='darkgrey',zorder=0,
                 alpha=0.3,edgecolor='none')

plt.yticks(np.arange(0,2,0.1),map(str,np.round(np.arange(0,2,0.1),2)),size=6)
plt.xticks(np.arange(1990,2030+1,10),map(str,np.arange(1990,2030+1,10)),size=6)
plt.xlim([1990,2020])   
plt.ylim([0,1])  

leg = plt.legend(shadow=False,fontsize=11,loc='upper center',
              bbox_to_anchor=(0.5,1.22),fancybox=True,ncol=4,frameon=False,
              handlelength=5,handletextpad=1)
for line,text in zip(leg.get_lines(), leg.get_texts()):
    text.set_color(line.get_color())

plt.text(1990,1.05,r'\textbf{[a]}',color='k',
         fontsize=7,ha='center')  
plt.ylabel(r'\textbf{Confidence}',color='k',fontsize=10)         

###############################################################################
ax = plt.subplot(212)

adjust_spines(ax, ['left', 'bottom'])
ax.spines['top'].set_color('none')
ax.spines['right'].set_color('none')
ax.spines['left'].set_color('dimgrey')
ax.spines['bottom'].set_color('dimgrey')
ax.spines['left'].set_linewidth(2)
ax.spines['bottom'].set_linewidth(2)
ax.tick_params('both',length=4,width=2,which='major',color='dimgrey')

rects = plt.bar(yearsall,actual_obs)
plt.plot(yearsall,predict_obs,linewidth=2,color='maroon',alpha=2,zorder=3,clip_on=False,
         linestyle='--',dashes=(1,0.3))
for i in range(len(rects)):
    rects[i].set_color('maroon')
    rects[i].set_edgecolor('w')
    rects[i].set_alpha(0.4)
    
plt.fill_between(x=yearsall[-10:],y1=0,y2=1,facecolor='darkgrey',zorder=0,
             alpha=0.3,edgecolor='none')

plt.yticks(np.arange(0,2,1),map(str,np.round(np.arange(0,2,1),2)),size=6)
plt.xticks(np.arange(1990,2030+1,10),map(str,np.arange(1990,2030+1,10)),size=6)
plt.xlim([1990,2020])   
plt.ylim([0,1])  

plt.text(1990,0.9,r'\textbf{ACTUAL HIATUS}',fontsize=20,color='maroon',alpha=0.4)
plt.text(1990,0.8,r'\textbf{{PREDICTED HIATUS}',fontsize=20,color='maroon',alpha=1)
plt.text(1990,1.05,r'\textbf{[b]}',color='k',
          fontsize=7,ha='center')  
plt.ylabel(r'\textbf{Classification}',color='k',fontsize=10)      
        
plt.tight_layout()
plt.subplots_adjust(hspace=0.4)
if rm_ensemble_mean == True:
    plt.savefig(directoryfigure + 'Obs-EDA_ConfidencePredictions_rmENSEMBLEmean.png',dpi=300)
else:
    plt.savefig(directoryfigure + 'Obs-EDA_ConfidencePredictions.png',dpi=300)