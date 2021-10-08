"""
First version of ANN to test new ML model - accompanying LRP script

Notes      : Need to use older TF1.15 environment with innvestigate

Author     : Zachary M. Labe
Date       : 25 August 2021
Version    : 1 (mostly for testing)
"""

### Import packages
import sys
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
from mpl_toolkits.basemap import Basemap, addcyclic, shiftgrid
import palettable.cubehelix as cm
import cmocean as cmocean
import calc_Utilities as UT
import calc_dataFunctions as df
import calc_Stats as dSS
import calc_LRPclass as LRP
import innvestigate
from sklearn.metrics import accuracy_score
import keras.backend as Back
from keras.layers import Dense, Activation, Dropout
from keras import regularizers,optimizers,metrics,initializers
from keras.utils import to_categorical
from keras.models import Sequential


### Plotting defaults 
plt.rc('text',usetex=True)
plt.rc('font',**{'family':'sans-serif','sans-serif':['Avant Garde']})

###############################################################################
###############################################################################
###############################################################################
### Original model
def loadmodel(Xtrain,Xtest,Ytrain,Ytest,hidden,random_network_seed,random_segment_seed,n_epochs,batch_size,lr_here,ridgePenalty,actFun,input_shape,output_shape,vari_predict):
    ### Directory of saved models
    dirname = '/Users/zlabe/Documents/Research/GmstTrendPrediction/SavedModels/'
    
    print('----ANN Training: learning rate = '+str(lr_here)+'; activation = '+actFun+'; batch = '+str(batch_size) + '----')
    Back.clear_session()
    model = Sequential()

    ### Input layer
    model.add(Dense(hidden[0],input_shape=(input_shape,),
                    activation=actFun))

    ### Initialize other layers
    for layer in hidden[1:]:
        model.add(Dense(layer,activation=actFun))
            
        print('\nTHIS IS AN ANN!\n')

    #### Initialize output layer
    model.add(Dense(output_shape,activation=None))

    ### Add softmax layer at the end
    model.add(Activation('softmax'))
    
    ### Add weights from compiled model
    savename = 'ANNv2_'+vari_predict[0]+'_hiatus_' + actFun + '_L2_'+ str(ridgePenalty)+ '_LR_' + str(lr_here)+ '_Batch'+ str(batch_size)+ '_Iters' + str(n_epochs) + '_' + str(len(hidden)) + 'x' + str(hidden[0]) + '_SegSeed' + str(random_segment_seed) + '_NetSeed'+ str(random_network_seed) 
    if(rm_ensemble_mean==True):
        savename = savename + '_EnsembleMeanRemoved'  
    
    modelwrite = dirname + savename + '.h5'
    model.load_weights(modelwrite)
    
    return model

### Hyperparamters for files of the ANN model
rm_ensemble_mean = True

if rm_ensemble_mean == False:
    variq = 'T2M'
    vari_predict = ['OHC100']
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
    yearsall = np.arange(1990,2099+1,1)
else:
    print(ValueError('SOMETHING IS WRONG WITH DATA PROCESSING!'))
    sys.exit()
    
### Read in data
directorymodel = '/Users/zlabe/Documents/Research/GmstTrendPrediction/SavedModels/'
savename = 'ANNv2_'+vari_predict[0]+'_hiatus_' + actFun + '_L2_'+ str(ridgePenalty)+ '_LR_' + str(lr_here)+ '_Batch'+ str(batch_size)+ '_Iters' + str(n_epochs) + '_' + str(len(hidden)) + 'x' + str(hidden[0]) + '_SegSeed' + str(random_segment_seed) + '_NetSeed'+ str(random_network_seed) 
if(rm_ensemble_mean==True):
    savename = savename + '_EnsembleMeanRemoved'  

origdata = np.load(directorymodel + savename + '.npz')
Xtrain = origdata['Xtrain']
Ytrain = origdata['Ytrain']
Xtest = origdata['Xtest']
Ytest = origdata['Ytest']
XobsS = origdata['XobsS']
lats = origdata['lats']
lons = origdata['lons']
yearsall = np.arange(1990,2100+1,1)

### Standardize
XtrainS,XtestS,stdVals = dSS.standardize_data(Xtrain,Xtest)
Xmean, Xstd = stdVals  

### ANN model paramaters
input_shape=np.shape(Xtrain)[1]
output_shape=np.shape(Ytrain)[1]

### Load model 
model = loadmodel(Xtrain,Xtest,Ytrain,Ytest,hidden,random_network_seed,random_segment_seed,n_epochs,batch_size,lr_here,ridgePenalty,actFun,input_shape,output_shape,vari_predict)

##############################################################################
##############################################################################
##############################################################################
### Calculate LRP
biasBool = False
num_of_class = 2
annType = 'class'
lrpRule = 'z'
normLRP = True
numLats = lats.shape[0]
numLons = lons.shape[0]
numDim = 3

lrpall = LRP.calc_LRPModel(model,np.append(XtrainS,XtestS,axis=0),
                                        np.append(Ytrain,Ytest,axis=0),
                                        biasBool,annType,num_of_class,
                                        yearsall,lrpRule,normLRP,
                                        numLats,numLons,numDim)
meanlrp = np.nanmean(lrpall,axis=0)
fig=plt.figure()
plt.contourf(meanlrp,300,cmap=cmocean.cm.thermal)

### For training data only
lrptrain = LRP.calc_LRPModel(model,XtrainS,Ytrain,biasBool,
                                        annType,num_of_class,
                                        yearsall,lrpRule,normLRP,
                                        numLats,numLons,numDim)

### For training data only
lrptest = LRP.calc_LRPModel(model,XtestS,Ytest,biasBool,
                                        annType,num_of_class,
                                        yearsall,lrpRule,normLRP,
                                        numLats,numLons,numDim)

### For observations data only
lrpobservations = LRP.calc_LRPObs(model,XobsS,biasBool,annType,
                                    num_of_class,yearsall,lrpRule,
                                    normLRP,numLats,numLons,numDim)

##############################################################################
##############################################################################
##############################################################################
def netcdfLRP(lats,lons,var,directory,typemodel,saveData):
    print('\n>>> Using netcdfLRP function!')
    
    from netCDF4 import Dataset
    import numpy as np
    
    name = 'LRPMap' + typemodel + '_' + saveData + '.nc'
    filename = directory + name
    ncfile = Dataset(filename,'w',format='NETCDF4')
    ncfile.description = 'LRP maps for using selected seed' 
    
    ### Dimensions
    ncfile.createDimension('years',var.shape[0])
    ncfile.createDimension('lat',var.shape[1])
    ncfile.createDimension('lon',var.shape[2])
    
    ### Variables
    years = ncfile.createVariable('years','f4',('years'))
    latitude = ncfile.createVariable('lat','f4',('lat'))
    longitude = ncfile.createVariable('lon','f4',('lon'))
    varns = ncfile.createVariable('LRP','f4',('years','lat','lon'))
    
    ### Units
    varns.units = 'unitless relevance'
    ncfile.title = 'LRP relevance'
    ncfile.instituion = 'Colorado State University'
    ncfile.references = 'Barnes et al. [2020]'
    
    ### Data
    years[:] = np.arange(var.shape[0])
    latitude[:] = lats
    longitude[:] = lons
    varns[:] = var
    
    ncfile.close()
    print('*Completed: Created netCDF4 File!')

directoryoutput = '/Users/zlabe/Documents/Research/GmstTrendPrediction/Data/'
netcdfLRP(lats,lons,lrpall,directoryoutput,'AllData',savename)
netcdfLRP(lats,lons,lrptrain,directoryoutput,'Training',savename)
netcdfLRP(lats,lons,lrptest,directoryoutput,'Testing',savename)
netcdfLRP(lats,lons,lrpobservations,directoryoutput,'Obs',savename)
