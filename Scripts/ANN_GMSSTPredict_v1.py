"""
Create an ANN to predict hiatus from only the GMSST

Author     : Zachary M. Labe
Date       : 5 January 2022
Version    : 1 (mostly for testing)
"""

### Import packages
import sys
import matplotlib.pyplot as plt
import numpy as np
import calc_Hiatus_v4 as HA
import calc_Utilities as UT
import calc_dataFunctions as df
import calc_SegmentData_v2 as FRAC
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.utils import to_categorical
import calc_Stats as dSS
from sklearn.metrics import accuracy_score,confusion_matrix,precision_recall_fscore_support,plot_confusion_matrix,precision_score,recall_score,f1_score

### Plotting defaults 
plt.rc('text',usetex=True)
plt.rc('font',**{'family':'sans-serif','sans-serif':['Avant Garde']})

###############################################################################
###############################################################################
###############################################################################
### Data preliminaries 
modelGCMs = ['CESM2le']
dataset_obs = 'ERA5'
allDataLabels = modelGCMs
monthlychoiceq = ['annual']
variables = ['T2M']
vari_predict = ['OHC100']
if vari_predict[0][:3] == 'OHC':
    obs_predict = 'OHC'
else:
    obs_predict = 'ERA5'
reg_name = 'SMILEGlobe'
level = 'surface'
ens1 = np.arange(1,10+1,1)
ens2 = np.arange(21,50+1,1)
ens = np.append(ens1,ens2)
###############################################################################
###############################################################################
randomalso = False
timeper = 'hiatus'
shuffletype = 'GAUSS'
###############################################################################
###############################################################################
land_only = False
ocean_only = False
###############################################################################
###############################################################################
baseline = np.arange(1951,1980+1,1)
###############################################################################
###############################################################################
window = 0
if window == 0:
    rm_standard_dev = False
    ravel_modelens = False
    ravelmodeltime = False
else:
    rm_standard_dev = True
    ravelmodeltime = False
    ravel_modelens = True
yearsall = np.arange(1979+window,2099+1,1)
yearsobs = np.arange(1979+window,2020+1,1)
###############################################################################
###############################################################################
numOfEns = 40
lentime = len(yearsall)
###############################################################################
###############################################################################
lat_bounds,lon_bounds = UT.regions(reg_name)
###############################################################################
###############################################################################
ravelyearsbinary = False
ravelbinary = False
lensalso = True
###############################################################################
###############################################################################
### Remove ensemble mean 
rm_ensemble_mean = True
###############################################################################
###############################################################################
### Call functions
trendlength = 10
AGWstart = 1990
years_newmodel = np.arange(AGWstart,yearsall[-1]+1,1)
years_newobs = np.arange(AGWstart,yearsobs[-1]+1,1)
vv = 0
mo = 0
variq = variables[vv]
monthlychoice = monthlychoiceq[mo]
saveData =  monthlychoice + '_' + variq + '_' + reg_name + '_' + dataset_obs
print('*Filename == < %s >' % saveData) 

### Parameters from ANN
variq = 'T2M'
fac = 1.0
random_segment_seed = int(np.genfromtxt('/Users/zlabe/Documents/Research/GmstTrendPrediction/Data/SelectedSegmentSeed.txt',unpack=True))
random_network_seed = 87750
hidden = [0]
n_epochs = 500
batch_size = 64
lr_here = 0.001
ridgePenalty = 0.00
actFun = 'linear'
fractWeight = 0.8
yearsall = np.arange(1990,2090+1,1)

### Naming conventions for files
directorymodel = '/Users/zlabe/Documents/Research/GmstTrendPrediction/SavedModels/'
savename = 'ANNv2_OHC100_hiatus_relu_L2_0.5_LR_0.001_Batch128_Iters500_2x30_SegSeed24120_NetSeed87750' 
if(rm_ensemble_mean==True):
    savename = savename + '_EnsembleMeanRemoved'   
    
### Directories to save files
directorydata = '/Users/zlabe/Documents/Research/GmstTrendPrediction/Data/'

###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
### Read in data for training predictions and actual hiatuses
trainindices = np.asarray(np.genfromtxt(directorydata + 'trainingEnsIndices_' + savename + '.txt'),dtype=int)
actual_train = np.genfromtxt(directorydata + 'trainingTrueLabels_' + savename + '.txt')

### Reshape arrays for [ensemble,year]
act_retrain = np.swapaxes(actual_train.reshape(trainindices.shape[0],1,yearsall.shape[0]),0,1).squeeze()

###############################################################################
###############################################################################
###############################################################################
### Read in data for testing predictions and actual hiatuses
testindices = np.asarray(np.genfromtxt(directorydata + 'testingEnsIndices_' + savename + '.txt'),dtype=int)
actual_test = np.genfromtxt(directorydata + 'testingTrueLabels_' + savename + '.txt')

### Reshape arrays for [ensemble,year]
act_retest = np.swapaxes(actual_test.reshape(testindices.shape[0],1,yearsall.shape[0]),0,1).squeeze()

###############################################################################
###############################################################################
###############################################################################
### Read in data for validation predictions and actual hiatuses
valindices = np.asarray(np.genfromtxt(directorydata + 'validationEnsIndices_' + savename + '.txt'),dtype=int)
actual_val = np.genfromtxt(directorydata + 'validationTrueLabels_' + savename + '.txt')

### Reshape arrays for [ensemble,year]
act_reval = np.swapaxes(actual_val.reshape(valindices.shape[0],1,yearsall.shape[0]),0,1).squeeze()

### Read in SST
SST = np.genfromtxt(directorydata + 'TimeSeries/GMSST_RM-ensmean_1990-2099.txt',unpack=True)
SST = np.genfromtxt(directorydata + 'TimeSeries/GMSST_1990-2099.txt',unpack=True)
SSTtrain = SST.transpose()[trainindices,:yearsall.shape[0]]
SSTtest = SST.transpose()[testindices,:yearsall.shape[0]]
SSTval= SST.transpose()[valindices,:yearsall.shape[0]]

###############################################################################
###############################################################################
###############################################################################
def loadmodel(Xtrain,Xval,Ytrain,Yval,hidden,random_network_seed,n_epochs,batch_size,lr_here,ridgePenalty,actFun,class_weight,input_shape,output_shape,fractWeight):
    print('----ANN Training: learning rate = '+str(lr_here)+'; activation = '+actFun+'; batch = '+str(batch_size) + '----')
    keras.backend.clear_session()
    model = keras.models.Sequential()
    
    ### Adjust class weights for too many hiatuses
    class_weight[1] = class_weight[1]*(fractWeight)

    #### One layer model
    model.add(Dense(output_shape,input_shape=(input_shape,),activation=actFun,use_bias=True,
                    kernel_regularizer=keras.regularizers.l1_l2(l1=0.00,l2=ridgePenalty),
                    bias_initializer=keras.initializers.RandomNormal(seed=random_network_seed),
                    kernel_initializer=keras.initializers.RandomNormal(seed=random_network_seed)))

    # ### Add softmax layer at the end
    model.add(Activation('softmax'))
    
    ### Compile the model
    model.compile(optimizer=keras.optimizers.SGD(lr=lr_here,
                  momentum=0.9,nesterov=True),  
                  loss = 'categorical_crossentropy',
                  metrics=[keras.metrics.categorical_accuracy]) 
    
    ### Callbacks
    early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss',
                                                   patience=10,
                                                   verbose=1,
                                                   mode='auto',
                                                   restore_best_weights=True)
    
    ### Model fit
    history = model.fit(Xtrain,Ytrain,batch_size=batch_size,epochs=n_epochs,
                        shuffle=True,verbose=1,class_weight=class_weight,
                        callbacks=[early_stopping],
                        validation_data=(Xval,Yval))
    
    ### See epochs
    plt.figure(figsize=(15,5))
    plt.subplot(1,2,1)
    plt.plot(history.history['loss'],label = 'training')
    plt.plot(history.history['val_loss'], label = 'validation')
    plt.title('loss')
    plt.xlabel('epoch')
    plt.legend()
    
    plt.subplot(1,2,2)
    plt.plot(history.history['categorical_accuracy'],label = 'training')
    plt.plot(history.history['val_categorical_accuracy'],label = 'validation')
    plt.title('Accuracy')
    plt.xlabel('epoch')
    plt.legend()
    
    model.summary() 
    return model,history

### Run ANN
np.random.seed(None) # 'unlock' the random seed

### Standardize
Xtrain = SSTtrain.reshape(trainindices.shape[0]*yearsall.shape[0],1)
Xval = SSTtest.reshape(valindices.shape[0]*yearsall.shape[0],1)
Xtest = SSTval.reshape(testindices.shape[0]*yearsall.shape[0],1)
XtrainS,XtestS,XvalS,stdVals = dSS.standardize_dataVal(Xtrain,Xtest,Xval)

Ytrain = act_retrain.reshape(trainindices.shape[0]*yearsall.shape[0])
Ytrain = to_categorical(Ytrain)
Yval = act_reval.reshape(valindices.shape[0]*yearsall.shape[0])
Yval= to_categorical(Yval)
Ytest = act_retest.reshape(testindices.shape[0]*yearsall.shape[0])
Ytest = to_categorical(Ytest)

### Set up output
def class_weight_creator(Y):
    class_dict = {}
    weights = np.max(np.sum(Y, axis=0)) / np.sum(Y, axis=0)
    for i in range(Y.shape[-1]):
        class_dict[i] = weights[i]               
    return class_dict
class_weight = class_weight_creator(Ytrain)
input_shape=np.shape(XtrainS)[1]
output_shape=np.shape(Ytrain)[1]
model,history = loadmodel(XtrainS,XvalS,Ytrain,Yval,hidden,random_network_seed,n_epochs,batch_size,lr_here,ridgePenalty,actFun,class_weight,input_shape,output_shape,fractWeight)

###############################################################################
###############################################################################
###############################################################################
### Prediction for training/testing
ypred_train = model.predict(XtrainS,verbose=1)
ypred_picktrain = np.argmax(ypred_train,axis=1)
ypred_test = model.predict(XtestS,verbose=1)
ypred_picktest = np.argmax(ypred_test,axis=1)

### Prediction for validation
ypred_val = model.predict(XvalS,verbose=1)
ypred_pickval = np.argmax(ypred_val,axis=1)

###############################################################################
###############################################################################
###############################################################################
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

acctrain = accuracyTotalTime(ypred_picktrain,actual_train)     
acctest = accuracyTotalTime(ypred_picktest,actual_test)
accval = accuracyTotalTime(ypred_pickval,actual_val)

prectrain = precisionTotalTime(ypred_picktrain,actual_train)     
prectest = precisionTotalTime(ypred_picktest,actual_test)
precval = precisionTotalTime(ypred_pickval,actual_val)

recalltrain = recallTotalTime(ypred_picktrain,actual_train)     
recalltest = recallTotalTime(ypred_picktest,actual_test)
recallval = recallTotalTime(ypred_pickval,actual_val)

f1_train = f1TotalTime(ypred_picktrain,actual_train)     
f1_test = f1TotalTime(ypred_picktest,actual_test)
f1_val = f1TotalTime(ypred_pickval,actual_val)
print('accuracy =',np.round(acctest,2),', precision =',np.round(prectest,2),
      ', recall =',np.round(recalltest,2),', F1 =',np.round(f1_test,2))

### Counts
print('\n')
print(np.unique(ypred_picktest, return_counts=True))
print(np.unique(actual_test, return_counts=True))

print(np.unique(ypred_picktrain, return_counts=True))
print(np.unique(actual_train, return_counts=True))