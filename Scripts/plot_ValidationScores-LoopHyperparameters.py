"""
Create plots to show validation scores for different hyperparameters

Author     : Zachary M. Labe
Date       : 28 September 2021
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
COUNTER = 5
hiddenall = [[10],[30],[10,10],[30,30],[10,10,10],[30,30,30]]
ridgePenaltyall = [0.01,0.1,0.5,1]
letters = ["a","b","c","d","e","f","g","h","i","j","k","l","m"]
combinations = COUNTER * len(hiddenall) * len(ridgePenaltyall)
directorydata = '/Users/zlabe/Documents/Research/GmstTrendPrediction/Data/LoopHyperparameters/'
directoryfigure = '/Users/zlabe/Desktop/GmstTrendPrediction/ANN_v2/Scores/'

### Read in hyperparameters
seeds = np.load(directorydata + 'Metadata_LoopResultsfor_ANNv2_OHC100_hiatus_EnsembleMeanRemoved_LoopHyper.npz')
random_segment_seedq = seeds['random_segment_seedall']
random_network_seedq = seeds['random_network_seedall']
savenameq = seeds['savenamesall']

accval = np.empty((combinations))
precval = np.empty((combinations))
recallval = np.empty((combinations))
f1val = np.empty((combinations))
for lo in range(combinations):
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
    savename = savenameq[lo]
        
    scores = np.load(directorydata + savename + '_SCORES_LoopHyper.npz')
    accval[lo] = scores['accval']
    precval[lo] = scores['precval']
    recallval[lo] = scores['recallval']
    f1val[lo] = scores['f1_val']
    
### Reshape arrays for plotting
acc_re = accval.reshape(len(hiddenall),len(ridgePenaltyall),COUNTER) * 100.
pre_re = precval.reshape(len(hiddenall),len(ridgePenaltyall),COUNTER) * 100.
rec_re = recallval.reshape(len(hiddenall),len(ridgePenaltyall),COUNTER) * 100.
f1s_re = f1val.reshape(len(hiddenall),len(ridgePenaltyall),COUNTER) * 100.

# ### First set of plots for layer and all l2
# acc_layer = acc_re.reshape(len(hiddenall),len(ridgePenaltyall)*COUNTER)
# pre_layer = pre_re.reshape(len(hiddenall),len(ridgePenaltyall)*COUNTER)
# rec_layer = rec_re.reshape(len(hiddenall),len(ridgePenaltyall)*COUNTER)
# f1s_layer = f1s_re.reshape(len(hiddenall),len(ridgePenaltyall)*COUNTER)

###############################################################################
###############################################################################
###############################################################################
### Graph for accuracy
labels = [r'\textbf{1-LAYER$_{10}$}',r'\textbf{1-LAYER$_{30}$}',r'\textbf{2-LAYERS$_{10}$}',
          r'\textbf{2-LAYERS$_{30}$}',r'\textbf{3-LAYERS$_{10}$}',r'\textbf{3-LAYERS$_{30}$}']

fig = plt.figure()
for plo in range(len(hiddenall)):
    ax = plt.subplot(2,3,plo+1)
    
    plotdata = acc_re[plo].transpose()
    
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
        plt.setp(bp['medians'], color='w',linewidth=1)
    
    positionsq = np.arange(len(ridgePenaltyall))
    bpl = plt.boxplot(plotdata,positions=positionsq,widths=0.6,
                      patch_artist=True,sym='')
    
    # Modify boxes
    cp= 'maroon'
    set_box_color(bpl,cp)
    plt.plot([], c=cp, label=r'\textbf{ACCURACY}',clip_on=False)
        
    for i in range(plotdata.shape[1]):
        y = plotdata[:,i]
        x = np.random.normal(positionsq[i], 0.04, size=len(y))
        plt.plot(x, y,color='teal', alpha=0.5,zorder=10,marker='.',linewidth=0,markersize=5,markeredgewidth=0,clip_on=False)
     
    if any([plo==0,plo==3]):
        plt.yticks(np.arange(0,101,10),list(map(str,np.round(np.arange(0,101,10),2))),
                    fontsize=6) 
        plt.ylim([80,100])
    else:
        plt.yticks(np.arange(0,101,10),list(map(str,np.round(np.arange(0,101,10),2))),
                    fontsize=6) 
        plt.ylim([80,100])
        ax.axes.yaxis.set_ticklabels([])
    
    if plo==3:
        plt.text(-0.35,78,r'\textbf{L$_{2}$=%s}' % ridgePenaltyall[0],fontsize=5,color='dimgrey',
                  ha='left',va='center')
        plt.text(1.05,78,r'\textbf{L$_{2}$=%s}' % ridgePenaltyall[1],fontsize=5,color='dimgrey',
                  ha='center',va='center')
        plt.text(2.4,78,r'\textbf{L$_{2}$=%s}' % ridgePenaltyall[2],fontsize=5,color='k',
                  ha='right',va='center')
        plt.text(3.27,78,r'\textbf{L$_{2}$=%s}' % ridgePenaltyall[3],fontsize=5,color='dimgrey',
                  ha='right',va='center')
    elif any([plo==4,plo==5]):
        plt.text(-0.35,78,r'\textbf{L$_{2}$=%s}' % ridgePenaltyall[0],fontsize=5,color='dimgrey',
                  ha='left',va='center')
        plt.text(1.05,78,r'\textbf{L$_{2}$=%s}' % ridgePenaltyall[1],fontsize=5,color='dimgrey',
                  ha='center',va='center')
        plt.text(2.4,78,r'\textbf{L$_{2}$=%s}' % ridgePenaltyall[2],fontsize=5,color='dimgrey',
                  ha='right',va='center')
        plt.text(3.27,78,r'\textbf{L$_{2}$=%s}' % ridgePenaltyall[3],fontsize=5,color='dimgrey',
                  ha='right',va='center')
        
    plt.text(1.65,98.5,r'%s' % labels[plo],fontsize=11,color='dimgrey',
              ha='center',va='center')
    plt.text(3.3,100.7,r'\textbf{[%s]}' % letters[plo],color='k',fontsize=6)
    
    if any([plo==0,plo==3]):
        plt.ylabel(r'\textbf{Accuracy [\%]}',color='k',fontsize=7)

if rm_ensemble_mean == True:
    plt.savefig(directoryfigure + 'AccracyScores-LoopHyperparameters_Hiatus_EDA-v2_rmENSEMBLEmean.png',dpi=300)
else:
    plt.savefig(directoryfigure + 'AccuracyScores-LoopHyperparameters_Hiatus_EDA-v2.png',dpi=300)
    
###############################################################################
###############################################################################
###############################################################################
### Graph for precision
    
fig = plt.figure()
for plo in range(len(hiddenall)):
    ax = plt.subplot(2,3,plo+1)
    
    plotdata = pre_re[plo].transpose()
    
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
        plt.setp(bp['medians'], color='w',linewidth=1)
    
    positionsq = np.arange(len(ridgePenaltyall))
    bpl = plt.boxplot(plotdata,positions=positionsq,widths=0.6,
                      patch_artist=True,sym='')
    
    # Modify boxes
    cp= 'maroon'
    set_box_color(bpl,cp)
    plt.plot([], c=cp, label=r'\textbf{PRECISION}',clip_on=False)
        
    for i in range(plotdata.shape[1]):
        y = plotdata[:,i]
        x = np.random.normal(positionsq[i], 0.04, size=len(y))
        plt.plot(x, y,color='teal', alpha=0.5,zorder=10,marker='.',linewidth=0,markersize=5,markeredgewidth=0,clip_on=False)
     
    if any([plo==0,plo==3]):
        plt.yticks(np.arange(0,101,10),list(map(str,np.round(np.arange(0,101,10),2))),
                    fontsize=6) 
        plt.ylim([0,60])
    else:
        plt.yticks(np.arange(0,101,10),list(map(str,np.round(np.arange(0,101,10),2))),
                    fontsize=6) 
        plt.ylim([0,60])
        ax.axes.yaxis.set_ticklabels([])
    
    if plo==3:
        plt.text(-0.35,-6,r'\textbf{L$_{2}$=%s}' % ridgePenaltyall[0],fontsize=5,color='dimgrey',
                  ha='left',va='center')
        plt.text(1.05,-6,r'\textbf{L$_{2}$=%s}' % ridgePenaltyall[1],fontsize=5,color='dimgrey',
                  ha='center',va='center')
        plt.text(2.4,-6,r'\textbf{L$_{2}$=%s}' % ridgePenaltyall[2],fontsize=5,color='k',
                  ha='right',va='center')
        plt.text(3.27,-6,r'\textbf{L$_{2}$=%s}' % ridgePenaltyall[3],fontsize=5,color='dimgrey',
                  ha='right',va='center')
    elif any([plo==4,plo==5]):
        plt.text(-0.35,-6,r'\textbf{L$_{2}$=%s}' % ridgePenaltyall[0],fontsize=5,color='dimgrey',
                  ha='left',va='center')
        plt.text(1.05,-6,r'\textbf{L$_{2}$=%s}' % ridgePenaltyall[1],fontsize=5,color='dimgrey',
                  ha='center',va='center')
        plt.text(2.4,-6,r'\textbf{L$_{2}$=%s}' % ridgePenaltyall[2],fontsize=5,color='dimgrey',
                  ha='right',va='center')
        plt.text(3.27,-6,r'\textbf{L$_{2}$=%s}' % ridgePenaltyall[3],fontsize=5,color='dimgrey',
                  ha='right',va='center')
        
    plt.text(1.65,55.3,r'%s' % labels[plo],fontsize=11,color='dimgrey',
              ha='center',va='center')
    plt.text(3.3,62,r'\textbf{[%s]}' % letters[plo],color='k',fontsize=6)
    
    if any([plo==0,plo==3]):
        plt.ylabel(r'\textbf{Precision [\%]}',color='k',fontsize=7)

if rm_ensemble_mean == True:
    plt.savefig(directoryfigure + 'PrecisionScores-LoopHyperparameters_Hiatus_EDA-v2_rmENSEMBLEmean.png',dpi=300)
else:
    plt.savefig(directoryfigure + 'PrecisionScores-LoopHyperparameters_Hiatus_EDA-v2.png',dpi=300)
    
###############################################################################
###############################################################################
###############################################################################
### Graph for recall
    
fig = plt.figure()
for plo in range(len(hiddenall)):
    ax = plt.subplot(2,3,plo+1)
    
    plotdata = rec_re[plo].transpose()
    
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
        plt.setp(bp['medians'], color='w',linewidth=1)
    
    positionsq = np.arange(len(ridgePenaltyall))
    bpl = plt.boxplot(plotdata,positions=positionsq,widths=0.6,
                      patch_artist=True,sym='')
    
    # Modify boxes
    cp= 'maroon'
    set_box_color(bpl,cp)
    plt.plot([], c=cp, label=r'\textbf{RECALL}',clip_on=False)
        
    for i in range(plotdata.shape[1]):
        y = plotdata[:,i]
        x = np.random.normal(positionsq[i], 0.04, size=len(y))
        plt.plot(x, y,color='teal', alpha=0.5,zorder=10,marker='.',linewidth=0,markersize=5,markeredgewidth=0,clip_on=False)
     
    if any([plo==0,plo==3]):
        plt.yticks(np.arange(0,101,10),list(map(str,np.round(np.arange(0,101,10),2))),
                    fontsize=6) 
        plt.ylim([0,60])
    else:
        plt.yticks(np.arange(0,101,10),list(map(str,np.round(np.arange(0,101,10),2))),
                    fontsize=6) 
        plt.ylim([0,60])
        ax.axes.yaxis.set_ticklabels([])
    
    if plo==3:
        plt.text(-0.35,-6,r'\textbf{L$_{2}$=%s}' % ridgePenaltyall[0],fontsize=5,color='dimgrey',
                  ha='left',va='center')
        plt.text(1.05,-6,r'\textbf{L$_{2}$=%s}' % ridgePenaltyall[1],fontsize=5,color='dimgrey',
                  ha='center',va='center')
        plt.text(2.4,-6,r'\textbf{L$_{2}$=%s}' % ridgePenaltyall[2],fontsize=5,color='k',
                  ha='right',va='center')
        plt.text(3.27,-6,r'\textbf{L$_{2}$=%s}' % ridgePenaltyall[3],fontsize=5,color='dimgrey',
                  ha='right',va='center')
    elif any([plo==4,plo==5]):
        plt.text(-0.35,-6,r'\textbf{L$_{2}$=%s}' % ridgePenaltyall[0],fontsize=5,color='dimgrey',
                  ha='left',va='center')
        plt.text(1.05,-6,r'\textbf{L$_{2}$=%s}' % ridgePenaltyall[1],fontsize=5,color='dimgrey',
                  ha='center',va='center')
        plt.text(2.4,-6,r'\textbf{L$_{2}$=%s}' % ridgePenaltyall[2],fontsize=5,color='dimgrey',
                  ha='right',va='center')
        plt.text(3.27,-6,r'\textbf{L$_{2}$=%s}' % ridgePenaltyall[3],fontsize=5,color='dimgrey',
                  ha='right',va='center')
        
    plt.text(1.65,55.3,r'%s' % labels[plo],fontsize=11,color='dimgrey',
              ha='center',va='center')
    plt.text(3.3,62,r'\textbf{[%s]}' % letters[plo],color='k',fontsize=6)
    
    if any([plo==0,plo==3]):
        plt.ylabel(r'\textbf{Recall [\%]}',color='k',fontsize=7)

if rm_ensemble_mean == True:
    plt.savefig(directoryfigure + 'RecallScores-LoopHyperparameters_Hiatus_EDA-v2_rmENSEMBLEmean.png',dpi=300)
else:
    plt.savefig(directoryfigure + 'RecallScores-LoopHyperparameters_Hiatus_EDA-v2.png',dpi=300)
    
###############################################################################
###############################################################################
###############################################################################
### Graph for f1-score
    
fig = plt.figure()
for plo in range(len(hiddenall)):
    ax = plt.subplot(2,3,plo+1)
    
    plotdata = f1s_re[plo].transpose()
    
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
        plt.setp(bp['medians'], color='w',linewidth=1)
    
    positionsq = np.arange(len(ridgePenaltyall))
    bpl = plt.boxplot(plotdata,positions=positionsq,widths=0.6,
                      patch_artist=True,sym='')
    
    # Modify boxes
    cp= 'maroon'
    set_box_color(bpl,cp)
    plt.plot([], c=cp, label=r'\textbf{F1-SCORE}',clip_on=False)
        
    for i in range(plotdata.shape[1]):
        y = plotdata[:,i]
        x = np.random.normal(positionsq[i], 0.04, size=len(y))
        plt.plot(x, y,color='teal', alpha=0.5,zorder=10,marker='.',linewidth=0,markersize=5,markeredgewidth=0,clip_on=False)
     
    if any([plo==0,plo==3]):
        plt.yticks(np.arange(0,101,10),list(map(str,np.round(np.arange(0,101,10),2))),
                    fontsize=6) 
        plt.ylim([0,60])
    else:
        plt.yticks(np.arange(0,101,10),list(map(str,np.round(np.arange(0,101,10),2))),
                    fontsize=6) 
        plt.ylim([0,60])
        ax.axes.yaxis.set_ticklabels([])
    
    if plo==3:
        plt.text(-0.35,-6,r'\textbf{L$_{2}$=%s}' % ridgePenaltyall[0],fontsize=5,color='dimgrey',
                  ha='left',va='center')
        plt.text(1.05,-6,r'\textbf{L$_{2}$=%s}' % ridgePenaltyall[1],fontsize=5,color='dimgrey',
                  ha='center',va='center')
        plt.text(2.4,-6,r'\textbf{L$_{2}$=%s}' % ridgePenaltyall[2],fontsize=5,color='k',
                  ha='right',va='center')
        plt.text(3.27,-6,r'\textbf{L$_{2}$=%s}' % ridgePenaltyall[3],fontsize=5,color='dimgrey',
                  ha='right',va='center')
    elif any([plo==4,plo==5]):
        plt.text(-0.35,-6,r'\textbf{L$_{2}$=%s}' % ridgePenaltyall[0],fontsize=5,color='dimgrey',
                  ha='left',va='center')
        plt.text(1.05,-6,r'\textbf{L$_{2}$=%s}' % ridgePenaltyall[1],fontsize=5,color='dimgrey',
                  ha='center',va='center')
        plt.text(2.4,-6,r'\textbf{L$_{2}$=%s}' % ridgePenaltyall[2],fontsize=5,color='dimgrey',
                  ha='right',va='center')
        plt.text(3.27,-6,r'\textbf{L$_{2}$=%s}' % ridgePenaltyall[3],fontsize=5,color='dimgrey',
                  ha='right',va='center')
        
    plt.text(1.65,55.3,r'%s' % labels[plo],fontsize=11,color='dimgrey',
              ha='center',va='center')
    plt.text(3.3,62,r'\textbf{[%s]}' % letters[plo],color='k',fontsize=6)
    
    if any([plo==0,plo==3]):
        plt.ylabel(r'\textbf{F1-Score [\%]}',color='k',fontsize=7)

if rm_ensemble_mean == True:
    plt.savefig(directoryfigure + 'F1Scores-LoopHyperparameters_Hiatus_EDA-v2_rmENSEMBLEmean.png',dpi=300)
else:
    plt.savefig(directoryfigure + 'F1Scores-LoopHyperparameters_Hiatus_EDA-v2.png',dpi=300)
    
###############################################################################
###############################################################################
###############################################################################
### Graph for combining recall and precision
    
fig = plt.figure()
for plo in range(len(hiddenall)):
    ax = plt.subplot(2,3,plo+1)
    
    plotdata1 = pre_re[plo].transpose()
    plotdata2 = rec_re[plo].transpose()
    
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
        plt.setp(bp['medians'], color='w',linewidth=1)
        
    positionspre = np.array(range(len(ridgePenaltyall)))*2.0-0.3
    positionspost = np.array(range(len(ridgePenaltyall)))*2.0+0.3
    bpl = plt.boxplot(plotdata1,positions=positionspre,widths=0.5,
                      patch_artist=True,sym='')
    bpr = plt.boxplot(plotdata2,positions=positionspost, widths=0.5,
                      patch_artist=True,sym='')
    
    # Modify boxes
    cpree = 'deepskyblue'
    cpost = 'indianred'
    set_box_color(bpl,cpree)
    set_box_color(bpr,cpost)
    plt.plot([], c=cpree, label=r'\textbf{PRECISION}')
    plt.plot([], c=cpost, label=r'\textbf{RECALL}')
    if plo == 2:
        l = plt.legend(shadow=False,fontsize=11,loc='upper center',
                      bbox_to_anchor=(-0.77,1.34),fancybox=True,ncol=4,frameon=False,
                      handlelength=5,handletextpad=1)
        for line,text in zip(l.get_lines(), l.get_texts()):
            text.set_color(line.get_color())
        
    for i in range(len(ridgePenaltyall)):
        y = plotdata1[:,i]
        x = np.random.normal(positionspre[i], 0.04, size=len(y))
        plt.plot(x, y,color='darkblue', alpha=0.5,zorder=10,marker='.',linewidth=0,markersize=5,markeredgewidth=0)
    for i in range(len(ridgePenaltyall)):
        y = plotdata2[:,i]
        x = np.random.normal(positionspost[i], 0.04, size=len(y))
        plt.plot(x, y,color='darkred', alpha=0.5,zorder=10,marker='.',linewidth=0,markersize=5,markeredgewidth=0)
     
    if any([plo==0,plo==3]):
        plt.yticks(np.arange(0,101,10),list(map(str,np.round(np.arange(0,101,10),2))),
                    fontsize=6) 
        plt.ylim([0,60])
    else:
        plt.yticks(np.arange(0,101,10),list(map(str,np.round(np.arange(0,101,10),2))),
                    fontsize=6) 
        plt.ylim([0,60])
        ax.axes.yaxis.set_ticklabels([])
    
    if plo==3:
        plt.text(-0.7,-6,r'\textbf{L$_{2}$=%s}' % ridgePenaltyall[0],fontsize=5,color='dimgrey',
                  ha='left',va='center')
        plt.text(2.1,-6,r'\textbf{L$_{2}$=%s}' % ridgePenaltyall[1],fontsize=5,color='dimgrey',
                  ha='center',va='center')
        plt.text(4.7,-6,r'\textbf{L$_{2}$=%s}' % ridgePenaltyall[2],fontsize=5,color='k',
                  ha='right',va='center')
        plt.text(6.5,-6,r'\textbf{L$_{2}$=%s}' % ridgePenaltyall[3],fontsize=5,color='dimgrey',
                  ha='right',va='center')
    elif any([plo==4,plo==5]):
        plt.text(-0.7,-6,r'\textbf{L$_{2}$=%s}' % ridgePenaltyall[0],fontsize=5,color='dimgrey',
                  ha='left',va='center')
        plt.text(2.1,-6,r'\textbf{L$_{2}$=%s}' % ridgePenaltyall[1],fontsize=5,color='dimgrey',
                  ha='center',va='center')
        plt.text(4.7,-6,r'\textbf{L$_{2}$=%s}' % ridgePenaltyall[2],fontsize=5,color='dimgrey',
                  ha='right',va='center')
        plt.text(6.5,-6,r'\textbf{L$_{2}$=%s}' % ridgePenaltyall[3],fontsize=5,color='dimgrey',
                  ha='right',va='center')
        
    plt.text(3.35,55.3,r'%s' % labels[plo],fontsize=11,color='dimgrey',
             ha='center',va='center')
    plt.text(6.4,62,r'\textbf{[%s]}' % letters[plo],color='k',fontsize=6)
    
    if any([plo==0,plo==3]):
        plt.ylabel(r'\textbf{Scores [\%]}',color='k',fontsize=7)

if rm_ensemble_mean == True:
    plt.savefig(directoryfigure + 'CombinedScores-LoopHyperparameters_Hiatus_EDA-v2_rmENSEMBLEmean.png',dpi=300)
else:
    plt.savefig(directoryfigure + 'CombinedScores-LoopHyperparameters_Hiatus_EDA-v2.png',dpi=300)