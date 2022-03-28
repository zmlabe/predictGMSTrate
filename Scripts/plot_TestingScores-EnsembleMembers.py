"""
Second version of ANN to test new ML model

Author     : Zachary M. Labe
Date       : 12 January 2022
Version    : 2 (mostly for testing; now using validation data, test number of
                ensemble members needed for training)
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
vari_predict = ['OHC100']
rm_ensemble_mean = True
COUNTER = 10
trainN_list = [2,7,12,17,22,27,32,37]
testN_list = [2,2,2,2,2,2,2,2]
valN_list = [1,1,1,1,1,1,1,1]
EXPERIMENTS = [5,10,15,20,25,30,35,40]
letters = ["a","b","c","d","e","f","g","h","i","j","k","l","m"]
combinations = COUNTER * len(EXPERIMENTS)
directorydata = '/Users/zlabe/Documents/Research/GmstTrendPrediction/Data/LoopEnsembles/'
directoryfigure = '/Users/zlabe/Desktop/GmstTrendPrediction/ANN_v2/Scores/'

### Read in hyperparameters
savenameFILE = 'LoopEnsembleResultsfor_ANNv2_'+vari_predict[0]+'_hiatus_'
seeds = np.load(directorydata + savenameFILE + '_SEEDS.npz')
random_segment_seedq = seeds['random_segment_seedall']
random_network_seedq = seeds['random_network_seedall']
savenameq = seeds['savenamesall']

       
scores = np.load(directorydata + savenameFILE + '_SCORES.npz')
acctest = scores['acctest'] * 100.
prectest = scores['prectest'] * 100.
recalltest = scores['recalltest'] * 100.
f1test = scores['f1_test'] * 100.

###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
### Graph for scores
fig = plt.figure()
ax = plt.subplot(111)

plotdata = f1test.transpose()
plotdata2 = acctest.transpose()

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
    plt.setp(bp['boxes'],color='w')
    plt.setp(bp['whiskers'], color='w',linewidth=1.5)
    plt.setp(bp['caps'], color='w',alpha=0)
    plt.setp(bp['medians'], color=color,linewidth=3)

positionsq = np.array(range(f1test.shape[0]))
bpl = plt.boxplot(plotdata,positions=positionsq,widths=0.6,
                  patch_artist=True,sym='',zorder=1)

# Modify boxes
cp= 'darkred'
set_box_color(bpl,cp)
plt.plot([], c=cp, label=r'\textbf{F1-SCORE}',clip_on=False)
leg = plt.legend(shadow=False,fontsize=11,loc='upper center',
              bbox_to_anchor=(0.5,1.14),fancybox=True,ncol=4,frameon=False,
              handlelength=5,handletextpad=1)
for line,text in zip(leg.get_lines(), leg.get_texts()):
    text.set_color(line.get_color())
    
for i in range(plotdata.shape[1]):
    y = plotdata[:,i]
    x = np.random.normal(positionsq[i], 0.04, size=len(y))
    plt.plot(x, y,color='darkred', alpha=0.6,zorder=10,marker='.',linewidth=0,markersize=10,markeredgewidth=0,clip_on=False)
  
###############################################################################
positionsq2 = np.array(range(acctest.shape[0]))
bpl2 = plt.boxplot(plotdata2,positions=positionsq2,widths=0.6,
                  patch_artist=True,sym='',zorder=1)

# Modify boxes
cp2= 'deepskyblue'
set_box_color(bpl2,cp2)
plt.plot([], c=cp2, label=r'\textbf{ACCURACY}',clip_on=False)
leg = plt.legend(shadow=False,fontsize=11,loc='upper center',
              bbox_to_anchor=(0.5,1.14),fancybox=True,ncol=4,frameon=False,
              handlelength=5,handletextpad=1)
for line,text in zip(leg.get_lines(), leg.get_texts()):
    text.set_color(line.get_color())
    
for i in range(plotdata2.shape[1]):
    y2 = plotdata2[:,i]
    x2 = np.random.normal(positionsq2[i], 0.04, size=len(y))
    plt.plot(x2, y2,color='darkblue', alpha=0.6,zorder=10,marker='.',linewidth=0,markersize=10,markeredgewidth=0,clip_on=False)
    
plt.yticks(np.arange(0,101,10),list(map(str,np.round(np.arange(0,101,10),2))),
            fontsize=8) 
plt.ylim([0,100])

plt.ylabel(r'\textbf{Scores [\%]}',color='k',fontsize=10)
plt.xlabel(r'\textbf{Number of Training Ensemble Members}',color='k',fontsize=10,labelpad=15)

plt.text(0,-5.4,r'%s' % trainN_list[0],fontsize=8,color='dimgrey',
          ha='center',va='center')
plt.text(1.0,-5.4,r'%s' % trainN_list[1],fontsize=8,color='dimgrey',
          ha='center',va='center')
plt.text(2,-5.4,r'%s' % trainN_list[2],fontsize=8,color='dimgrey',
          ha='center',va='center')
plt.text(3,-5.4,r'%s' % trainN_list[3],fontsize=8,color='dimgrey',
          ha='center',va='center')
plt.text(4,-5.4,r'%s' % trainN_list[4],fontsize=8,color='dimgrey',
          ha='center',va='center')
plt.text(5.0,-5.4,r'%s' % trainN_list[5],fontsize=8,color='dimgrey',
          ha='center',va='center')
plt.text(6,-5.4,r'%s' % trainN_list[6],fontsize=8,color='dimgrey',
          ha='center',va='center')
plt.text(7,-5.4,r'%s' % trainN_list[7],fontsize=8,color='dimgrey',
          ha='center',va='center')

if rm_ensemble_mean == True:
    plt.savefig(directoryfigure + 'TestingScores-LoopEnsembleMembers_Hiatus_EDA-v2_rmENSEMBLEmean.png',dpi=900)
else:
    plt.savefig(directoryfigure + 'TestingScores-LoopEnsembleMembers_Hiatus_EDA-v2.png',dpi=300)