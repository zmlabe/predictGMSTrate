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
directoryfigureTRAIN = '/Users/zlabe/Desktop/GmstTrendPrediction/ANN_v2/Training/'
directoryfigureTEST = '/Users/zlabe/Desktop/GmstTrendPrediction/ANN_v2/Testing/'

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
### Try training data for confusion matrix
cm_train = confusion_matrix(actual_train,predict_train) 
cm_train_normtrue = confusion_matrix(actual_train,predict_train,normalize='true')
cm_train_normpred = confusion_matrix(actual_train,predict_train,normalize='pred')
plot_cmtrainNORM = np.flipud(cm_train_normtrue)
plot_cmtrainPRED = np.flipud(cm_train_normpred)
plot_cmtrain = np.flipud(cm_train)

### Calculate baseline
numhiatustrain_actual = np.sum(cm_train,axis=1)[1]
numCCtrain_actual = np.sum(cm_train,axis=1)[0]
total_actualtrain = numhiatustrain_actual + numCCtrain_actual
baseline_hiatustrain = np.round((numhiatustrain_actual/total_actualtrain)*100,1)
baseline_CCtrain = np.round((numCCtrain_actual/total_actualtrain)*100,1)
baseline_train = np.array([[baseline_CCtrain,baseline_hiatustrain],[baseline_CCtrain,baseline_hiatustrain]])

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
cs = plt.pcolormesh(plot_cmtrain,shading='faceted',edgecolor='w',
                    linewidth=7,cmap=csm,vmin=0,vmax=3100)

ylabels = [r'\textbf{Hiatus [n=%s]}' % (np.sum(cm_train,axis=1)[1]),r'\textbf{Climate Change [n=%s]}' % (np.sum(cm_train,axis=1)[0])]
plt.yticks(np.arange(0.5,2.5,1),ylabels,ha='center',color='dimgrey',
           rotation=90,va='center',size=15)
yax = ax.get_yaxis()
yax.set_tick_params(pad=20)
xlabels = [r'\textbf{Climate Change [n=%s]}' % (np.sum(cm_train,axis=0)[0]),r'\textbf{Hiatus [n=%s]}' % (np.sum(cm_train,axis=0)[1])]
plt.xticks(np.arange(0.5,2.5,1),xlabels,ha='center',color='dimgrey',size=15)
xax = ax.get_xaxis()
xax.set_tick_params(pad=15)

for i in range(plot_cmtrain.shape[0]):
    for j in range(plot_cmtrain.shape[1]):          
        plt.text(j+0.5,i+0.5,r'\textbf{%s}' % plot_cmtrain[i,j],fontsize=50,
            color='crimson',va='center',ha='center')
        plt.text(j+0.5,i+0.34,r'\textbf{Recall = %s \%% }' % (np.round(plot_cmtrainNORM[i,j]*100,1)),fontsize=10,
            color='dimgrey',va='center',ha='center')
        plt.text(j+0.5,i+0.29,r'\textbf{Precision =  %s \%% }' % (np.round(plot_cmtrainPRED[i,j]*100,1)),fontsize=10,
            color='k',va='center',ha='center')
        if any([(i==0) & (j==1),(i==1) & (j==0)]):
            plt.text(j+0.5,i+0.24,r'\textbf{[ %s \%% ]}' % (baseline_train[i,j]),fontsize=10,
                color='crimson',va='center',ha='center',alpha=0.5)
        
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
cbar.set_label(r'\textbf{TRAINING [Accuracy = %s \%%, Recall = %s \%%, Precision = %s \%%]}' % ((np.round(acctrain,3)*100),(np.round(recalltrain,3)*100),
               (np.round(prectrain,3)*100)),color='k',labelpad=10,fontsize=15)

plt.tight_layout()
if rm_ensemble_mean == True:
    plt.savefig(directoryfigureTRAIN + 'ConfusionMatrix_Train_Hiatus_EDA-v1_rmENSEMBLEmean.png',dpi=300)
else:
    plt.savefig(directoryfigureTRAIN + 'ConfusionMatrix_Train_Hiatus_EDA-v1.png',dpi=300)

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
cs = plt.pcolormesh(plot_cmtest,shading='faceted',edgecolor='w',
                    linewidth=7,cmap=csm,vmin=0,vmax=900)

ylabels = [r'\textbf{Slowdown [n=%s]}' % (np.sum(cm_test,axis=1)[1]),r'\textbf{Decadal Warming [n=%s]}' % (np.sum(cm_test,axis=1)[0])]
plt.yticks(np.arange(0.5,2.5,1),ylabels,ha='center',color='dimgrey',
           rotation=90,va='center',size=15)
yax = ax.get_yaxis()
yax.set_tick_params(pad=20)
xlabels = [r'\textbf{Decadal Warming [n=%s]}' % (np.sum(cm_test,axis=0)[0]),r'\textbf{Slowdown [n=%s]}' % (np.sum(cm_test,axis=0)[1])]
plt.xticks(np.arange(0.5,2.5,1),xlabels,ha='center',color='dimgrey',size=15)
xax = ax.get_xaxis()
xax.set_tick_params(pad=15)

for i in range(plot_cmtest.shape[0]):
    for j in range(plot_cmtest.shape[1]):          
        plt.text(j+0.5,i+0.5,r'\textbf{%s}' % plot_cmtest[i,j],fontsize=50,
            color='crimson',va='center',ha='center')
        plt.text(j+0.5,i+0.34,r'\textbf{Recall = %s \%% }' % (np.round(plot_cmtestNORM[i,j]*100,1)),fontsize=10,
            color='k',va='center',ha='center')
        plt.text(j+0.5,i+0.29,r'\textbf{Precision = %s \%% }' % (np.round(plot_cmtestPRED[i,j]*100,1)),fontsize=10,
            color='k',va='center',ha='center')
        if any([(i==0) & (j==1),(i==1) & (j==0)]):
            plt.text(j+0.5,i+0.24,r'\textbf{[ %s \%% ]}' % (baseline_test[i,j]),fontsize=10,
                color='crimson',va='center',ha='center',alpha=0.5)
        
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
cbar.set_label(r'\textbf{TESTING DATA}',color='k',labelpad=10,fontsize=15)

plt.tight_layout()
if rm_ensemble_mean == True:
    plt.savefig(directoryfigureTEST + 'ConfusionMatrix_Test_Hiatus_EDA-v1_rmENSEMBLEmean.png',dpi=300)
else:
    plt.savefig(directoryfigureTEST + 'ConfusionMatrix_Test_Hiatus_EDA-v1.png',dpi=300)
