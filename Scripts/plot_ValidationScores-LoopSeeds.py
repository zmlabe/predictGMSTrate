"""
Create plots to show validation scores for different seeds

Author     : Zachary M. Labe
Date       : 27 September 2021
Version    : 2 (mostly for testing)
"""

### Import packages
import sys
import matplotlib.pyplot as plt
import matplotlib.colors as c
import numpy as np

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

accval = np.empty((COUNTER))
precval = np.empty((COUNTER))
recallval = np.empty((COUNTER))
f1val = np.empty((COUNTER))
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
        yearsall = np.arange(1990,2099+1,1)
    else:
        print(ValueError('SOMETHING IS WRONG WITH DATA PROCESSING!'))
        sys.exit()
    
    ### Naming conventions for files
    savename = 'LoopSeedsResultsfor_ANNv2_'+vari_predict[0]+'_hiatus_' + actFun + '_L2_'+ str(ridgePenalty)+ '_LR_' + str(lr_here)+ '_Batch'+ str(batch_size)+ '_Iters' + str(n_epochs) + '_' + str(len(hidden)) + 'x' + str(hidden[0]) + '_SegSeed' + str(random_segment_seed) + '_NetSeed'+ str(random_network_seed)
    
    if(rm_ensemble_mean==True):
        savename = savename + '_EnsembleMeanRemoved'  
        
    scores = np.load(directorydata + savename + '_SCORES_%s.npz' % lo)
    accval[lo] = scores['accval']
    precval[lo] = scores['precval']
    recallval[lo] = scores['recallval']
    f1val[lo] = scores['f1_val']
    
### Gather data and place percent
alldata = np.asarray([accval,precval,recallval,f1val]) * 100

###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
### Graph for scores
fig = plt.figure()
ax = plt.subplot(111)

plotdata = alldata.transpose()

adjust_spines(ax, ['left', 'bottom'])
ax.spines['top'].set_color('none')
ax.spines['right'].set_color('none')
ax.spines['left'].set_color('dimgrey')
ax.spines['bottom'].set_color('none')
ax.spines['left'].set_linewidth(2)
ax.tick_params('both',length=4,width=2,which='major',color='dimgrey')
ax.tick_params(axis="x",which="both",bottom = False,top=False,
                labelbottom=False)

ax.yaxis.grid(zorder=1,color='darkgrey',alpha=0.7,clip_on=False,linewidth=0.5)

def set_box_color(bp, color):
    plt.setp(bp['boxes'],color=color)
    plt.setp(bp['whiskers'], color=color,linewidth=1.5)
    plt.setp(bp['caps'], color='w',alpha=0)
    plt.setp(bp['medians'], color='w',linewidth=1.5)

positionsq = np.array(range(alldata.shape[0]))
bpl = plt.boxplot(plotdata,positions=positionsq,widths=0.6,
                  patch_artist=True,sym='')

# Modify boxes
cp= 'maroon'
set_box_color(bpl,cp)
plt.plot([], c=cp, label=r'\textbf{VALIDATION}',clip_on=False)
leg = plt.legend(shadow=False,fontsize=11,loc='upper center',
              bbox_to_anchor=(0.5,1.14),fancybox=True,ncol=4,frameon=False,
              handlelength=5,handletextpad=1)
for line,text in zip(leg.get_lines(), leg.get_texts()):
    text.set_color(line.get_color())
    
for i in range(plotdata.shape[1]):
    y = plotdata[:,i]
    x = np.random.normal(positionsq[i], 0.04, size=len(y))
    plt.plot(x, y,color='teal', alpha=0.5,zorder=10,marker='.',linewidth=0,markersize=5,markeredgewidth=0,clip_on=False)
    
plt.yticks(np.arange(0,101,10),list(map(str,np.round(np.arange(0,101,10),2))),
            fontsize=6) 
plt.ylim([10,90])

plt.text(-0.3,3,r'\textbf{ACCURACY}',fontsize=10,color='dimgrey',
          ha='left',va='center')
plt.text(1.,3,r'\textbf{PRECISION}',fontsize=10,color='dimgrey',
          ha='center',va='center')
plt.text(2.2,3,r'\textbf{RECALL}',fontsize=10,color='dimgrey',
          ha='right',va='center')
plt.text(3.27,3,r'\textbf{F1-SCORE}',fontsize=10,color='dimgrey',
          ha='right',va='center')

plt.ylabel(r'\textbf{Score [\%]}',color='k',fontsize=10)

if rm_ensemble_mean == True:
    plt.savefig(directoryfigure + 'ValidationScores-LoopSeeds_Hiatus_EDA-v2_rmENSEMBLEmean.png',dpi=300)
else:
    plt.savefig(directoryfigure + 'ValidationScores-LoopSeeds_Hiatus_EDA-v2.png',dpi=300)