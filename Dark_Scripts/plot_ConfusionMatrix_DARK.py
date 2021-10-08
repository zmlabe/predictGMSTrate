"""
First exploratory data analysis for making LRP plots of ANNv1

Author     : Zachary M. Labe
Date       : 2 September 2021
Version    : 1 (mostly for testing)
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
from sklearn.metrics import accuracy_score,confusion_matrix,precision_recall_fscore_support,plot_confusion_matrix,precision_score,recall_score

### Plotting defaults 
plt.rc('text',usetex=True)
plt.rc('font',**{'family':'sans-serif','sans-serif':['Avant Garde']}) 
plt.rc('savefig',facecolor='black')
plt.rc('axes',edgecolor='darkgrey')
plt.rc('xtick',color='darkgrey')
plt.rc('ytick',color='darkgrey')
plt.rc('axes',labelcolor='darkgrey')
plt.rc('axes',facecolor='black')

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
    yearsall = np.arange(1990,2099+1,1)
elif rm_ensemble_mean == True:
    vari_predict= ['OHC100']
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
    yearsall = np.arange(1990,2099+1,1)
else:
    print(ValueError('SOMETHING IS WRONG WITH DATA PROCESSING!'))
    sys.exit()

### Naming conventions for files
dirname = '/Users/zlabe/Documents/Research/GmstTrendPrediction/SavedModels/'
savename = 'ANNv2_'+vari_predict[0]+'_hiatus_' + actFun + '_L2_'+ str(ridgePenalty)+ '_LR_' + str(lr_here)+ '_Batch'+ str(batch_size)+ '_Iters' + str(n_epochs) + '_' + str(len(hidden)) + 'x' + str(hidden[0]) + '_SegSeed' + str(random_segment_seed) + '_NetSeed'+ str(random_network_seed) 

if(rm_ensemble_mean==True):
    savename = savename + '_EnsembleMeanRemoved' 

### Directories to save files
directorydata = '/Users/zlabe/Documents/Research/GmstTrendPrediction/Data/'
directoryfigureTEST = '/Users/zlabe/Documents/Projects/predictGMSTrate/Dark_Figures/'

###############################################################################
###############################################################################
###############################################################################
### Read in data for training predictions and actual hiatuses
trainindices = np.genfromtxt(directorydata + 'trainingEnsIndices_' + savename + '.txt')
actual_train = np.genfromtxt(directorydata + 'trainingTrueLabels_' + savename + '.txt')
predict_train = np.genfromtxt(directorydata + 'trainingPredictedLabels_' + savename+ '.txt')

###############################################################################
###############################################################################
###############################################################################
### Read in data for testing predictions and actual hiatuses
testindices = np.genfromtxt(directorydata + 'testingEnsIndices_' + savename + '.txt')
actual_test = np.genfromtxt(directorydata + 'testingTrueLabels_' + savename + '.txt')
predict_test = np.genfromtxt(directorydata + 'testingPredictedLabels_' + savename+ '.txt')

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
### Count hiatuses in testing
uniquetest,counttest = np.unique(predict_test,return_counts=True)
actual_uniquetest,actual_counttest = np.unique(actual_test ,return_counts=True)

##############################################################################
##############################################################################
##############################################################################   
### Calculate accuracy statistics
def accuracyTotalTime(data_pred,data_true):
    """
    Compute accuracy for the entire time series
    """  
    data_truer = data_true
    data_predr = data_pred
    accdata_pred = accuracy_score(data_truer,data_predr)
        
    return accdata_pred

def precisionTotalTime(data_pred,data_true):
    """
    Compute precision for the entire time series
    """
    data_truer = data_true
    data_predr = data_pred
    precdata_pred = precision_score(data_truer,data_predr)
    
    return precdata_pred

def recallTotalTime(data_pred,data_true):
    """
    Compute recall for the entire time series
    """
    data_truer = data_true
    data_predr = data_pred
    recalldata_pred = recall_score(data_truer,data_predr)
    
    return recalldata_pred

acctrain = accuracyTotalTime(predict_train,actual_train)     
acctest = accuracyTotalTime(predict_test,actual_test)
print('Accuracy Training == ',np.round(acctrain,3))
print('Accuracy Testing == ',np.round(acctest,3))

prectrain = precisionTotalTime(predict_train,actual_train)     
prectest = precisionTotalTime(predict_test,actual_test)

recalltrain = recallTotalTime(predict_train,actual_train)     
recalltest = recallTotalTime(predict_test,actual_test)

###############################################################################
###############################################################################
###############################################################################
### Try testing data for confusion matrix
cm_test = confusion_matrix(actual_test,predict_test) 
cm_test_normtrue = confusion_matrix(actual_test,predict_test,normalize='true')
cm_test_normpred = confusion_matrix(actual_test,predict_test,normalize='pred')
plot_cmtestNORM = np.flipud(cm_test_normtrue)
plot_cmtestPRED = np.flipud(cm_test_normpred)
plot_cmtest = np.flipud(cm_test)

### Calculate baseline
numhiatustest_actual = np.sum(cm_test,axis=1)[1]
numCCtest_actual = np.sum(cm_test,axis=1)[0]
total_actualtest = numhiatustest_actual + numCCtest_actual
baseline_hiatustest = np.round((numhiatustest_actual/total_actualtest)*100,1)
baseline_CCtest = np.round((numCCtest_actual/total_actualtest)*100,1)
baseline_test = np.array([[baseline_CCtest,baseline_hiatustest],[baseline_CCtest,baseline_hiatustest]])

fig = plt.figure(figsize=(8,8))
ax = plt.subplot(111)

ax.spines['top'].set_color('none')
ax.spines['right'].set_color('none')
ax.spines['bottom'].set_color('none')
ax.spines['left'].set_color('none')
ax.get_xaxis().set_tick_params(direction='out',width=2,length=3,
            color='darkgrey')
ax.get_yaxis().set_tick_params(direction='out',width=2,length=3,
            color='darkgrey')

plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom='on')
plt.tick_params(
    axis='y',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    left=False,      # ticks along the bottom edge are off
    right=False,         # ticks along the top edge are off
    labelleft='on')

csm=plt.get_cmap(cmr.ocean_r)
cs = plt.pcolormesh(plot_cmtest,shading='faceted',edgecolor='k',
                    linewidth=15,cmap=csm,vmin=0,vmax=900)

ylabels = [r'\textbf{Slowdown [n = %s]}' % (np.sum(cm_test,axis=1)[1]),r'\textbf{Other [n = %s]}' % (np.sum(cm_test,axis=1)[0])]
plt.yticks(np.arange(0.5,2.5,1),ylabels,ha='center',color='w',
           rotation=90,va='center',size=15)
yax = ax.get_yaxis()
yax.set_tick_params(pad=20)
xlabels = [r'\textbf{Other [n = %s]}' % (np.sum(cm_test,axis=0)[0]),r'\textbf{Slowdown [n = %s]}' % (np.sum(cm_test,axis=0)[1])]
plt.xticks(np.arange(0.5,2.5,1),xlabels,ha='center',color='w',size=15)
xax = ax.get_xaxis()
xax.set_tick_params(pad=15)

for i in range(plot_cmtest.shape[0]):
    for j in range(plot_cmtest.shape[1]):          
        plt.text(j+0.5,i+0.5,r'\textbf{%s}' % plot_cmtest[i,j],fontsize=50,
            color='crimson',va='center',ha='center')
        plt.text(j+0.5,i+0.34,r'\textbf{Recall = %s \%% }' % (np.round(plot_cmtestNORM[i,j]*100,1)),fontsize=12,
            color='dimgrey',va='center',ha='center')
        plt.text(j+0.5,i+0.29,r'\textbf{Precision = %s \%% }' % (np.round(plot_cmtestPRED[i,j]*100,1)),fontsize=12,
            color='k',va='center',ha='center')
        if any([(i==0) & (j==1),(i==1) & (j==0)]):
            plt.text(j+0.5,i+0.24,r'\textbf{[Baseline = %s \%% ]}' % (baseline_test[i,j]),fontsize=8,
                color='crimson',va='center',ha='center',alpha=0.7)
        
plt.text(1,-0.04,r'\textbf{PREDICTED}',color='crimson',fontsize=13,ha='center',
         va='center')
plt.text(-0.02,1,r'\textbf{ACTUAL}',color='crimson',fontsize=13,ha='center',
         va='center',rotation=90)

# cbar = plt.colorbar(cs,orientation='horizontal',aspect=50,pad=0.11)
# barlim = np.arange(0,1000,100)
# cbar.set_ticks(barlim)
# cbar.set_ticklabels(list(map(str,barlim)))  
# cbar.ax.tick_params(axis='x', size=.001,labelsize=7)
# cbar.outline.set_edgecolor('darkgrey')

plt.tight_layout()
if rm_ensemble_mean == True:
    plt.savefig(directoryfigureTEST + 'ConfusionMatrix_Test_DARK.png',dpi=300)
