"""
First version of ANN to test new ML model

Author     : Zachary M. Labe
Date       : 23 August 2021
Version    : 1 (mostly for testing)
"""

### Import packages
import sys
import matplotlib.pyplot as plt
import numpy as np
import calc_Hiatus_v3 as HA
import calc_Utilities as UT
import calc_dataFunctions as df
import calc_SegmentData as FRAC
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
from tensorflow.keras.layers import Dense, Activation
import calc_Stats as dSS

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
### Processing data steps
rm_ensemble_mean = True
###############################################################################
###############################################################################
###############################################################################
###############################################################################
### Read in model and observational/reanalysis data
readData = True
if readData == True:
    def read_primary_dataset(variq,dataset,monthlychoice,numOfEns,lensalso,randomalso,ravelyearsbinary,ravelbinary,shuffletype,timeper,lat_bounds=lat_bounds,lon_bounds=lon_bounds):
        data,lats,lons = df.readFiles(variq,dataset,monthlychoice,numOfEns,lensalso,randomalso,ravelyearsbinary,ravelbinary,shuffletype,timeper)
        datar,lats,lons = df.getRegion(data,lats,lons,lat_bounds,lon_bounds)
        print('\nOur dataset: ',dataset,' is shaped',data.shape)
        return datar,lats,lons
      
    def read_obs_dataset(variq,dataset_obs,numOfEns,lensalso,randomalso,ravelyearsbinary,ravelbinary,shuffletype,lat_bounds=lat_bounds,lon_bounds=lon_bounds):
        data_obs,lats_obs,lons_obs = df.readFiles(variq,dataset_obs,monthlychoice,numOfEns,lensalso,randomalso,ravelyearsbinary,ravelbinary,shuffletype,timeper)
        data_obs,lats_obs,lons_obs = df.getRegion(data_obs,lats_obs,lons_obs,lat_bounds,lon_bounds)
        
        print('our OBS dataset: ',dataset_obs,' is shaped',data_obs.shape)
        return data_obs,lats_obs,lons_obs
    
    ### Call functions
    trendlength = 10
    AGWstart = 1990
    years_newmodel = np.arange(AGWstart,yearsall[-1]+1,1)
    years_newobs = np.arange(AGWstart,yearsobs[-1]+1,1)
    vv = 0
    mo = 0
    variq = variables[vv]
    monthlychoice = monthlychoiceq[mo]
    directoryfigure = '/Users/zlabe/Desktop/GmstTrendPrediction/'
    saveData =  monthlychoice + '_' + variq + '_' + reg_name + '_' + dataset_obs
    print('*Filename == < %s >' % saveData) 
    
    ### Read data for hiatus periods
    models = []
    modelsm = []
    obs = []
    obsm = []
    SLOPEthreshh_o = []
    SLOPEthreshh_m = []
    diff_o = []
    diff_m = []
    yearstrend_obsh = []
    linetrend_obsh = []
    indexslopeNegative_obsh = []
    classes_obsh = []
    yearstrend_mh = []
    linetrend_mh = []
    indexslopeNegative_mh = []
    classes_mh = []
    count = []
    for i in range(len(modelGCMs)):
        dataset = modelGCMs[i]
        modelsq,lats,lons = read_primary_dataset(variq,dataset,monthlychoice,numOfEns,
                                                lensalso,randomalso,ravelyearsbinary,
                                                ravelbinary,shuffletype,timeper,
                                                lat_bounds,lon_bounds)
        obsq,lats,lons = read_obs_dataset(variq,dataset_obs,numOfEns,lensalso,randomalso,ravelyearsbinary,ravelbinary,shuffletype,lat_bounds=lat_bounds,lon_bounds=lon_bounds)
        
        ### Calculate global mean temperature
        lon2,lat2 = np.meshgrid(lons,lats)
        modelsmq = UT.calc_weightedAve(modelsq,lat2)
        obsmq = UT.calc_weightedAve(obsq,lat2)
        
        ### Calculate thresholds for hiatus period
        SLOPEthreshh_oq,diff_oq = HA.calc_thresholdOfTrend(obsmq,trendlength,yearsobs,AGWstart,'hiatus')
        SLOPEthreshh_mq,diff_mq = HA.calc_thresholdOfTrend(modelsmq,trendlength,yearsall,AGWstart,'hiatus')
        
        ### Calculate actual hiatus periods in climate models and observations
        yearstrend_obshq,linetrend_obshq,indexslopeNegative_obshq,classes_obshq = HA.calc_HiatusAcc(obsmq,trendlength,yearsobs,AGWstart,SLOPEthreshh_oq,'hiatus',diff_oq)
        yearstrend_mhq,linetrend_mhq,indexslopeNegative_mhq,classes_mhq = HA.calc_HiatusAcc(modelsmq,trendlength,yearsall,AGWstart,SLOPEthreshh_mq,'hiatus',diff_oq)
    
        ### County how many hiatus
        countq = len(indexslopeNegative_mhq)
    
        ### Save for each data set separately
        models.append(modelsq)
        modelsm.append(modelsmq)
        obs.append(obsq)
        obsm.append(obsmq)
        SLOPEthreshh_o.append(SLOPEthreshh_oq)
        SLOPEthreshh_m.append(SLOPEthreshh_mq)
        diff_o.append(diff_oq)
        diff_m.append(diff_mq)
        yearstrend_obsh.append(yearstrend_obshq)
        linetrend_obsh.append(linetrend_obshq)
        indexslopeNegative_obsh.append(indexslopeNegative_obshq)
        classes_obsh.append(classes_obshq)
        yearstrend_mh.append(yearstrend_mhq)
        linetrend_mh.append(linetrend_mhq)
        indexslopeNegative_mh.append(indexslopeNegative_mhq)
        classes_mh.append(classes_mhq)
        count.append(countq)
        
    ### Check for arrays
    models = np.asarray(models)
    modelsm = np.asarray(modelsm)
    obs = np.asarray(obs).squeeze()
    obsm = np.asarray(obsm).squeeze()
    SLOPEthreshh_o = np.asarray(SLOPEthreshh_o).squeeze()
    SLOPEthreshh_m = np.asarray(SLOPEthreshh_m)
    diff_o = np.asarray(diff_o).squeeze()
    diff_m = np.asarray(diff_m)
    yearstrend_obsh = np.asarray(yearstrend_obsh).squeeze()
    linetrend_obsh = np.asarray(linetrend_obsh).squeeze()
    indexslopeNegative_obsh = np.asarray(indexslopeNegative_obsh).squeeze()
    classes_obsh = np.asarray(classes_obsh).squeeze()
    yearstrend_mh = np.asarray(yearstrend_mh)
    linetrend_mh = np.asarray(linetrend_mh)
    indexslopeNegative_mh = np.asarray(indexslopeNegative_mh)
    classes_mh = np.asarray(classes_mh)
    count = np.asarray(count)
    
    ### Function to read in predictor variables (SST/OHC)
    models_var = []
    obs_var = []
    for i in range(len(modelGCMs)):
        dataset = modelGCMs[i]
        modelsq_var,lats,lons = read_primary_dataset(vari_predict[0],dataset,monthlychoice,numOfEns,
                                                lensalso,randomalso,ravelyearsbinary,
                                                ravelbinary,shuffletype,timeper,
                                                lat_bounds,lon_bounds)
        obsq_var,lats,lons = read_obs_dataset(vari_predict[0],obs_predict,numOfEns,lensalso,randomalso,ravelyearsbinary,ravelbinary,shuffletype,lat_bounds=lat_bounds,lon_bounds=lon_bounds)
        
        ### Save predictor
        models_var.append(modelsq_var)
        obs_var.append(obsq_var)
    models_var = np.asarray(models_var)
    obs_var = np.asarray(obs_var).squeeze()
    
    ### Remove ensemble mean
    if rm_ensemble_mean == True:
        models_var = dSS.remove_ensemble_mean(models_var,ravel_modelens,
                                              ravelmodeltime,rm_standard_dev,
                                              numOfEns)
        print('\n*Removed ensemble mean*')
        
    ### Slice for number of years
    yearsq_m = np.where((yearsall >= AGWstart))[0]
    yearsq_o = np.where((yearsobs >= AGWstart))[0]
    models_slice = models_var[:,:,yearsq_m,:,:]
    obs_slice = obs_var[yearsq_o,:,:]
    
    ### Missing land data to zeros
    models_slice[np.isnan(models_slice)] = 0.
    obs_slice[np.isnan(obs_slice)] = 0.

### Segment data for training and testing data
fac = 0.8
random_segment_seed = int(np.genfromtxt('/Users/zlabe/Documents/Research/GmstTrendPrediction/Data/SelectedSegmentSeed.txt',unpack=True))
random_network_seed = 87750
Xtrain,Ytrain,Xtest,Ytest,Xtrain_shape,Xtest_shape,testIndices,trainIndices,class_weight,random_segment_seed = FRAC.segment_data(models_slice,classes_mh,fac,random_segment_seed)

### Model paramaters
if rm_ensemble_mean == True:
    # OHC100
    hidden = [30,30]
    n_epochs = 500
    batch_size = 128
    lr_here = 0.001
    ridgePenalty = 0.35
    actFun = 'relu'
    fractWeight = 0.5
    input_shape=np.shape(Xtrain)[1]
    output_shape=np.shape(Ytrain)[1]
    
    # OHC700
    # hidden = [20,20]
    # n_epochs = 500
    # batch_size = 128
    # lr_here = 0.001
    # ridgePenalty = 0.3
    # actFun = 'relu'
    # fractWeight = 0.5
    # input_shape=np.shape(Xtrain)[1]
    # output_shape=np.shape(Ytrain)[1]
    
    # OHC300
    # hidden = [10,10]
    # n_epochs = 500
    # batch_size = 128
    # lr_here = 0.001
    # ridgePenalty = 0.15
    # actFun = 'relu'
    # fractWeight = 0.5
    # input_shape=np.shape(Xtrain)[1]
    # output_shape=np.shape(Ytrain)[1]
elif rm_ensemble_mean == False:
    hidden = [20,20]
    n_epochs = 500
    batch_size = 128
    lr_here = 0.001
    ridgePenalty = 0.05
    actFun = 'relu'
    fractWeight = 0.5
    input_shape=np.shape(Xtrain)[1]
    output_shape=np.shape(Ytrain)[1]
else:
    print(ValueError('SOMETHING IS WRONG WITH DATA PROCESSING!'))
    sys.exit()

def loadmodel(Xtrain,Xtest,Ytrain,Ytest,hidden,random_network_seed,n_epochs,batch_size,lr_here,ridgePenalty,actFun,class_weight,input_shape,output_shape,fractWeight):
    print('----ANN Training: learning rate = '+str(lr_here)+'; activation = '+actFun+'; batch = '+str(batch_size) + '----')
    keras.backend.clear_session()
    model = keras.models.Sequential()
    
    ### Adjust class weights for too many hiatuses
    class_weight[1] = class_weight[1]*(fractWeight)

    ### Input layer
    model.add(Dense(hidden[0],input_shape=(input_shape,),
                    activation=actFun,use_bias=True,
                    kernel_regularizer=keras.regularizers.l1_l2(l1=0.00,l2=ridgePenalty),
                    bias_initializer=keras.initializers.RandomNormal(seed=random_network_seed),
                    kernel_initializer=keras.initializers.RandomNormal(seed=random_network_seed)))

    ### Initialize other layers
    for layer in hidden[1:]:
        model.add(Dense(layer,activation=actFun,
                        use_bias=True,
                        kernel_regularizer=keras.regularizers.l1_l2(l1=0.00,l2=0.00),
                        bias_initializer=keras.initializers.RandomNormal(seed=random_network_seed),
                        kernel_initializer=keras.initializers.RandomNormal(seed=random_network_seed)))
         
        # model.add(layers.Dropout(rate=0.125,seed=random_network_seed)) 
        print('\nTHIS IS AN ANN!\n')

    #### Initialize output layer
    model.add(Dense(output_shape,activation=None,use_bias=True,
                    kernel_regularizer=keras.regularizers.l1_l2(l1=0.00, l2=0.00),
                    bias_initializer=keras.initializers.RandomNormal(seed=random_network_seed),
                    kernel_initializer=keras.initializers.RandomNormal(seed=random_network_seed)))

    ### Add softmax layer at the end
    model.add(Activation('softmax'))
    
    ### Compile the model
    model.compile(optimizer=keras.optimizers.SGD(lr=lr_here,
                  momentum=0.9,nesterov=True),  
                  loss = 'categorical_crossentropy',
                  metrics=[keras.metrics.categorical_accuracy]) 
    
    ### Callbacks
    early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss',
                                                   patience=5,
                                                   verbose=1,
                                                   mode='auto',
                                                   restore_best_weights=True)
    
    ### Model fit
    history = model.fit(Xtrain,Ytrain,batch_size=batch_size,epochs=n_epochs,
                        shuffle=True,verbose=1,class_weight=class_weight,
                        callbacks=[early_stopping],
                        validation_data=(Xtest,Ytest))
    
    ### See epochs
    plt.figure(figsize=(15,5))
    plt.subplot(1,2,1)
    plt.plot(history.history['loss'],label = 'training')
    plt.plot(history.history['val_loss'], label = 'testing')
    plt.title('loss')
    plt.xlabel('epoch')
    plt.legend()
    
    plt.subplot(1,2,2)
    plt.plot(history.history['categorical_accuracy'],label = 'training')
    plt.plot(history.history['val_categorical_accuracy'],label = 'testing')
    plt.title('Accuracy')
    plt.xlabel('epoch')
    plt.legend()
    
    model.summary() 
    return model,history

### Standardize data
XtrainS,XtestS,stdVals = dSS.standardize_data(Xtrain,Xtest)
Xmean, Xstd = stdVals  

### Compile neural network
model,history = loadmodel(XtrainS,XtestS,Ytrain,Ytest,hidden,random_network_seed,n_epochs,batch_size,lr_here,ridgePenalty,actFun,class_weight,input_shape,output_shape,fractWeight)

### Prediction for training/testing
ypred_train = model.predict(XtrainS)
ypred_picktrain = np.argmax(ypred_train,axis=1)
ypred_test = model.predict(XtestS)
ypred_picktest = np.argmax(ypred_test,axis=1)

### Actual hiatus
actual_classtrain = np.swapaxes(classes_mh,0,1)[trainIndices,:,:].ravel()
actual_classtest = np.swapaxes(classes_mh,0,1)[testIndices,:,:].ravel()

### Count hiatuses
uniquetrain,counttrain = np.unique(ypred_picktrain,return_counts=True)
uniquetest,counttest = np.unique(ypred_picktest,return_counts=True)
actual_uniquetrain,actual_counttrain = np.unique(actual_classtrain,return_counts=True)
actual_uniquetest,actual_counttest = np.unique(actual_classtest,return_counts=True)

### Test observations
obsreshape = obs_slice.reshape(obs_slice.shape[0],obs_slice.shape[1]*obs_slice.shape[2])
Xmeanobs = np.nanmean(obsreshape,axis=0)
Xstdobs = np.nanstd(obsreshape,axis=0)  
XobsS = (obsreshape-Xmeanobs)/Xstdobs
XobsS[np.isnan(XobsS)] = 0.

testobs = model.predict(XobsS)
selectobs = np.argmax(testobs,axis=1)
actualobs = np.array(classes_obsh,dtype=int)

plt.figure()
plt.plot(selectobs,alpha=0.5,linewidth=2)
plt.plot(actualobs,alpha=0.5,linewidth=2)

### Start saving everything, including the ANN
dirname = '/Users/zlabe/Documents/Research/GmstTrendPrediction/SavedModels/'
savename = 'ANN_'+variq+'_hiatus_' + actFun + '_L2_'+ str(ridgePenalty)+ '_LR_' + str(lr_here)+ '_Batch'+ str(batch_size)+ '_Iters' + str(n_epochs) + '_' + str(len(hidden)) + 'x' + str(hidden[0]) + '_SegSeed' + str(random_segment_seed) + '_NetSeed'+ str(random_network_seed) 

if(rm_ensemble_mean==True):
    savename = savename + '_EnsembleMeanRemoved'  
modelwrite = dirname + savename + '.h5'
model.save_weights(modelwrite)
np.savez(dirname + savename + '.npz',trainModels=trainIndices,testModels=testIndices,Xtrain=Xtrain,Ytrain=Ytrain,Xtest=Xtest,Ytest=Ytest,Xmean=Xmean,Xstd=Xstd,lats=lats,lons=lons,XobsS=XobsS)

### Observations saving output
directoryoutput = '/Users/zlabe/Documents/Research/GmstTrendPrediction/Data/'
np.savetxt(directoryoutput + 'obsLabels_' + savename + '.txt',selectobs)
np.savetxt(directoryoutput + 'obsActualLabels_' + savename + '.txt',actualobs)
np.savetxt(directoryoutput + 'obsConfid_' + savename + '.txt',testobs)

## Training/testing for saving output
np.savetxt(directoryoutput + 'trainingEnsIndices_' + savename + '.txt',trainIndices)
np.savetxt(directoryoutput + 'testingEnsIndices_' + savename + '.txt',testIndices)

np.savetxt(directoryoutput + 'trainingTrueLabels_' + savename + '.txt',actual_classtrain)
np.savetxt(directoryoutput + 'testingTrueLabels_' + savename + '.txt',actual_classtest)

np.savetxt(directoryoutput + 'trainingPredictedLabels_' + savename + '.txt',ypred_picktrain)
np.savetxt(directoryoutput + 'trainingPredictedConfidence_' + savename+ '.txt',ypred_train)
np.savetxt(directoryoutput + 'testingPredictedLabels_' + savename+ '.txt',ypred_picktest)
np.savetxt(directoryoutput + 'testingPredictedConfidence_' + savename+ '.txt',ypred_test)
