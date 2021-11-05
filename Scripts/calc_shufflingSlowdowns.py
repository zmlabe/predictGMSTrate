"""
Monte-carlo shuffling of slowdown events from uniform distributions

Author     : Zachary M. Labe
Date       : 5 November 2021
Version    : 1
"""

### Import packages
import sys
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score,confusion_matrix,precision_recall_fscore_support,plot_confusion_matrix,precision_score,recall_score,f1_score

### Plotting defaults 
plt.rc('text',usetex=True)
plt.rc('font',**{'family':'sans-serif','sans-serif':['Avant Garde']})

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

def f1TotalTime(data_pred,data_true):
    """
    Compute f1 for the entire time series
    """
    data_truer = data_true
    data_predr = data_pred
    f1data_pred = f1_score(data_truer,data_predr)
    
    return f1data_pred

### By chance fraction
slowdown = 63
warming  = 543
n = slowdown + warming
chance = slowdown/n

### Calculate slowdowns
size = 10000 
fillcount = np.empty((size))
acc = np.empty((size))
pre = np.empty((size))
rec = np.empty((size))
f1score = np.empty((size))
for i in range(size):
    inputdata = np.random.uniform(low=0.0,high=1.0,size=1000)
    inputdata[np.where(inputdata < (1.0-chance))] = 0
    inputdata[np.where(inputdata > 0)] = 1
    
    prediction = np.random.uniform(low=0.0,high=1.0,size=1000)
    prediction[np.where(prediction < (1.0-chance))] = 0
    prediction[np.where(prediction > 0)] = 1
    
    ### Scores
    acc[i] = accuracyTotalTime(prediction,inputdata)
    pre[i] = precisionTotalTime(prediction,inputdata)
    rec[i] = recallTotalTime(prediction,inputdata)
    f1score[i] = f1TotalTime(prediction,inputdata)

    countslow = np.count_nonzero(inputdata==1)
    countwarming = np.count_nonzero(inputdata==0)
    fillcount[i] = countslow/(countwarming+countslow)
    
meandist = np.nanmean(fillcount)
meanacc = np.nanmean(acc)
meanpre = np.nanmean(pre)
meanrec = np.nanmean(rec)
meanf1 = np.nanmean(f1score)
print('%s random chance - ANN' % (chance))
print('%s random chance - distribution' % (meandist))
print(meanacc,meanpre,meanrec,meanf1)
plt.hist(acc)
plt.hist(f1score)