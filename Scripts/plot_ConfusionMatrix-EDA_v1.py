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
from sklearn.metrics import accuracy_score,confusion_matrix,precision_recall_fscore_support,plot_confusion_matrix

### Plotting defaults 
plt.rc('text',usetex=True)
plt.rc('font',**{'family':'sans-serif','sans-serif':['Avant Garde']})

### Hyperparamters for files of the ANN model
rm_ensemble_mean = False

if rm_ensemble_mean == False:
    variq = 'T2M'
    fac = 0.8
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
    variq = 'T2M'
    fac = 0.8
    random_segment_seed = int(np.genfromtxt('/Users/zlabe/Documents/Research/GmstTrendPrediction/Data/SelectedSegmentSeed.txt',unpack=True))
    random_network_seed = 87750
    hidden = [30,30]
    n_epochs = 500
    batch_size = 128
    lr_here = 0.001
    ridgePenalty = 0.35
    actFun = 'relu'
    fractWeight = 0.5
    yearsall = np.arange(1990,2099+1,1)
else:
    print(ValueError('SOMETHING IS WRONG WITH DATA PROCESSING!'))
    sys.exit()

### Naming conventions for files
directorymodel = '/Users/zlabe/Documents/Research/GmstTrendPrediction/SavedModels/'
savename = 'ANN_'+variq+'_hiatus_' + actFun + '_L2_'+ str(ridgePenalty)+ '_LR_' + str(lr_here)+ '_Batch'+ str(batch_size)+ '_Iters' + str(n_epochs) + '_' + str(len(hidden)) + 'x' + str(hidden[0]) + '_SegSeed' + str(random_segment_seed) + '_NetSeed'+ str(random_network_seed)
if(rm_ensemble_mean==True):
    savename = savename + '_EnsembleMeanRemoved'  

### Directories to save files
directorydata = '/Users/zlabe/Documents/Research/GmstTrendPrediction/Data/'
directoryfigureTRAIN = '/Users/zlabe/Desktop/GmstTrendPrediction/ANN_v1/Training/'
directoryfigureTEST = '/Users/zlabe/Desktop/GmstTrendPrediction/ANN_v1/Testing/'

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
     
acctest = accuracyTotalTime(predict_test,actual_test)
print('Accuracy Testing == ',np.round(acctest,3))

###############################################################################
###############################################################################
###############################################################################
### Try training data for confusion matrix
cm_train = confusion_matrix(actual_train,predict_train) 
cm_train_norm = confusion_matrix(actual_train,predict_train,normalize='true')
plot_cmtrainNORM = np.flipud(cm_train_norm)
plot_cmtrain = np.flipud(cm_train)

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
    bottom='on',      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom='on')
plt.tick_params(
    axis='y',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    left='on',      # ticks along the bottom edge are off
    right=False,         # ticks along the top edge are off
    labelleft='on')

csm=plt.get_cmap(cmr.ocean_r)
cs = plt.pcolormesh(plot_cmtrain,shading='faceted',edgecolor='w',
                    linewidth=3,cmap=csm,vmin=0,vmax=3100)

ylabels = [r'\textbf{Hiatus}',r'\textbf{Climate Change}']
plt.yticks(np.arange(0.5,2.5,1),ylabels,ha='center',color='dimgrey',
           rotation=90,va='center',size=15)
yax = ax.get_yaxis()
yax.set_tick_params(pad=20)
xlabels = [r'\textbf{Climate Change}',r'\textbf{Hiatus}']
plt.xticks(np.arange(0.5,2.5,1),xlabels,ha='center',color='dimgrey',size=15)
xax = ax.get_xaxis()
xax.set_tick_params(pad=15)

for i in range(plot_cmtrain.shape[0]):
    for j in range(plot_cmtrain.shape[1]):          
        plt.text(j+0.5,i+0.5,r'\textbf{%s}' % plot_cmtrain[i,j],fontsize=45,
            color='crimson',va='center',ha='center')
        plt.text(j+0.5,i+0.34,r'\textbf{[ %s \%% ]}' % (np.round(plot_cmtrainNORM[i,j],3)*100),fontsize=10,
            color='dimgrey',va='center',ha='center')
        
plt.text(1,-0.04,r'\textbf{PREDICTED}',color='k',fontsize=10,ha='center',
         va='center')
plt.text(-0.02,1,r'\textbf{ACTUAL}',color='k',fontsize=10,ha='center',
         va='center',rotation=90)

cbar = plt.colorbar(cs,orientation='horizontal',aspect=50,pad=0.11)
barlim = np.arange(0,3101,500)
cbar.set_ticks(barlim)
cbar.set_ticklabels(list(map(str,barlim)))  
cbar.ax.tick_params(axis='x', size=.001,labelsize=7)
cbar.outline.set_edgecolor('darkgrey')
cbar.set_label(r'\textbf{TRAINING DATA}',color='k',labelpad=10,fontsize=20)

plt.tight_layout()
plt.savefig(directoryfigureTRAIN + 'ConfusionMatrix_Train_Hiatus_EDA-v1.png',dpi=300)

### Try testing data for confusion matrix
cm_test = confusion_matrix(actual_test,predict_test) 
cm_test_norm = confusion_matrix(actual_test,predict_test,normalize='true')
plot_cmtestNORM = np.flipud(cm_test_norm)
plot_cmtest = np.flipud(cm_test)

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
    bottom='on',      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom='on')
plt.tick_params(
    axis='y',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    left='on',      # ticks along the bottom edge are off
    right=False,         # ticks along the top edge are off
    labelleft='on')

csm=plt.get_cmap(cmr.ocean_r)
cs = plt.pcolormesh(plot_cmtest,shading='faceted',edgecolor='w',
                    linewidth=3,cmap=csm,vmin=0,vmax=900)

ylabels = [r'\textbf{Hiatus}',r'\textbf{Climate Change}']
plt.yticks(np.arange(0.5,2.5,1),ylabels,ha='center',color='dimgrey',
           rotation=90,va='center',size=15)
yax = ax.get_yaxis()
yax.set_tick_params(pad=20)
xlabels = [r'\textbf{Climate Change}',r'\textbf{Hiatus}']
plt.xticks(np.arange(0.5,2.5,1),xlabels,ha='center',color='dimgrey',size=15)
xax = ax.get_xaxis()
xax.set_tick_params(pad=15)

for i in range(plot_cmtest.shape[0]):
    for j in range(plot_cmtest.shape[1]):          
        plt.text(j+0.5,i+0.5,r'\textbf{%s}' % plot_cmtest[i,j],fontsize=45,
            color='crimson',va='center',ha='center')
        plt.text(j+0.5,i+0.34,r'\textbf{[ %s \%% ]}' % (np.round(plot_cmtestNORM[i,j],3)*100),fontsize=10,
            color='dimgrey',va='center',ha='center')
        
plt.text(1,-0.04,r'\textbf{PREDICTED}',color='k',fontsize=10,ha='center',
         va='center')
plt.text(-0.02,1,r'\textbf{ACTUAL}',color='k',fontsize=10,ha='center',
         va='center',rotation=90)

cbar = plt.colorbar(cs,orientation='horizontal',aspect=50,pad=0.11)
barlim = np.arange(0,1000,100)
cbar.set_ticks(barlim)
cbar.set_ticklabels(list(map(str,barlim)))  
cbar.ax.tick_params(axis='x', size=.001,labelsize=7)
cbar.outline.set_edgecolor('darkgrey')
cbar.set_label(r'\textbf{TESTING DATA}',color='k',labelpad=10,fontsize=20)

plt.tight_layout()
plt.savefig(directoryfigureTEST + 'ConfusionMatrix_Test_Hiatus_EDA-v1.png',dpi=300)
