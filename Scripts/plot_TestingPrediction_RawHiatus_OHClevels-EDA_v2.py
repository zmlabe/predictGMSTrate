"""
Explore raw composites based on indices from predicted testing data and
showing all the difference OHC levels

Author     : Zachary M. Labe
Date       : 21 September 2021
Version    : 2 (mostly for testing)
"""

### Import packages
import sys
import matplotlib.pyplot as plt
import numpy as np
import calc_Utilities as UT
from mpl_toolkits.basemap import Basemap, addcyclic, shiftgrid
import palettable.cubehelix as cm
import cmocean as cmocean
import calc_dataFunctions as df
import calc_Stats as dSS
from netCDF4 import Dataset

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
vari_predict = ['SST','OHC100','OHC300','OHC700']
obs_predict = 'OHC'
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
### Remove ensemble mean 
rm_ensemble_mean = True
###############################################################################
###############################################################################
### Accuracy for composites
accurate = 'WRONG'
if accurate == True:
    typemodel = 'correcthiatus'
elif accurate == False:
    typemodel = 'extrahiatus'
elif accurate == 'WRONG':
    typemodel = 'wronghiatus'
elif accurate == 'HIATUS':
    typemodel = 'allhiatus'
###############################################################################
###############################################################################
### Call functions
trendlength = 10
AGWstart = 1990
years_newmodel = np.arange(AGWstart,yearsall[-1]-8,1)
years_newobs = np.arange(AGWstart,yearsobs[-1]-8,1)
vv = 0
mo = 0
variq = variables[vv]
monthlychoice = monthlychoiceq[mo]
directoryfigure = '/Users/zlabe/Desktop/GmstTrendPrediction/ANN_v2/Raw/'
saveData =  monthlychoice + '_' + variq + '_' + reg_name + '_' + dataset_obs
print('*Filename == < %s >' % saveData) 

###############################################################################
###############################################################################
### Function to read in predictor variables (SST/OHC)
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

###############################################################################
###############################################################################
### Loop through to read all the variables
ohcHIATUS = np.empty((len(vari_predict),92,144))
for vvv in range(len(vari_predict)):
    models_var = []
    for i in range(len(modelGCMs)):
        dataset = modelGCMs[i]
        modelsq_var,lats,lons = read_primary_dataset(vari_predict[vvv],dataset,monthlychoice,numOfEns,
                                                lensalso,randomalso,ravelyearsbinary,
                                                ravelbinary,shuffletype,timeper,
                                                lat_bounds,lon_bounds)
        
        ### Save predictor
        models_var.append(modelsq_var)
    models_var = np.asarray(models_var)
    
    ### Remove ensemble mean
    if rm_ensemble_mean == True:
        models_var = dSS.remove_ensemble_mean(models_var,ravel_modelens,
                                              ravelmodeltime,rm_standard_dev,
                                              numOfEns)
        print('\n*Removed ensemble mean*')
    
    ### Standardize
    models_varravel = models_var.squeeze().reshape(numOfEns*yearsall.shape[0],lats.shape[0]*lons.shape[0])
    meanvar = np.nanmean(models_varravel,axis=0)
    stdvar = np.nanstd(models_varravel,axis=0)
    modelsstd_varravel = (models_varravel-meanvar)/stdvar
    models_var = modelsstd_varravel.reshape(len(modelGCMs),numOfEns,yearsall.shape[0],lats.shape[0],lons.shape[0])
        
    ### Slice for number of years
    yearsq_m = np.where((yearsall >= AGWstart))[0]
    models_slice = models_var[:,:,yearsq_m,:,:]
    
    if rm_ensemble_mean == False:
        variq = 'T2M'
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
    elif rm_ensemble_mean == True:
        variq = 'T2M'
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
    else:
        print(ValueError('SOMETHING IS WRONG WITH DATA PROCESSING!'))
        sys.exit()
    
    ### Naming conventions for files
    directorymodel = '/Users/zlabe/Documents/Research/GmstTrendPrediction/SavedModels/'
    savename = 'ANNv2_'+'OHC100'+'_hiatus_' + actFun + '_L2_'+ str(ridgePenalty)+ '_LR_' + str(lr_here)+ '_Batch'+ str(batch_size)+ '_Iters' + str(n_epochs) + '_' + str(len(hidden)) + 'x' + str(hidden[0]) + '_SegSeed' + str(random_segment_seed) + '_NetSeed'+ str(random_network_seed) 
    if(rm_ensemble_mean==True):
        savename = savename + '_EnsembleMeanRemoved'   
        
    ### Directories to save files
    directorydata = '/Users/zlabe/Documents/Research/GmstTrendPrediction/Data/'
    
    ###############################################################################
    ###############################################################################
    ###############################################################################
    ### Read in data for testing predictions and actual hiatuses
    testindices = np.asarray(np.genfromtxt(directorydata + 'testingEnsIndices_' + savename + '.txt'),dtype=int)
    actual_test = np.genfromtxt(directorydata + 'testingTrueLabels_' + savename + '.txt')
    predict_test = np.genfromtxt(directorydata + 'testingPredictedLabels_' + savename+ '.txt')
    
    ### Reshape arrays for [ensemble,year]
    act_re = np.swapaxes(actual_test.reshape(testindices.shape[0],1,years_newmodel.shape[0]),0,1).squeeze()
    pre_re = np.swapaxes(predict_test.reshape(testindices.shape[0],1,years_newmodel.shape[0]),0,1).squeeze()
    
    ### Slice ensembles for testing data
    ohcready = models_slice[:,testindices,:act_re.shape[1],:,:].squeeze()
    
    ### Pick all hiatuses
    if accurate == True: ### correct predictions
        ohc_allenscomp = []
        for ens in range(ohcready.shape[0]):
            ohc_comp = []
            for yr in range(ohcready.shape[1]):
                if (pre_re[ens,yr]) == 1 and (act_re[ens,yr] == 1):
                    ohc_comp.append(ohcready[ens,yr,:,:])
            ohc_allenscomp.append(ohc_comp)
    elif accurate == False: ### picks all hiatus predictions
        ohc_allenscomp = []
        for ens in range(ohcready.shape[0]):
            ohc_comp = []
            for yr in range(ohcready.shape[1]):
                if pre_re[ens,yr] == 1:
                    ohc_comp.append(ohcready[ens,yr,:,:])
            ohc_allenscomp.append(ohc_comp)
    elif accurate == 'WRONG': ### picks hiatus but is wrong
        ohc_allenscomp = []
        for ens in range(ohcready.shape[0]):
            ohc_comp = []
            for yr in range(ohcready.shape[1]):
                if (pre_re[ens,yr]) == 1 and (act_re[ens,yr] == 0):
                    ohc_comp.append(ohcready[ens,yr,:,:])
            ohc_allenscomp.append(ohc_comp)
    elif accurate == 'HIATUS': ### accurate climate change
        ohc_allenscomp = []
        for ens in range(ohcready.shape[0]):
            ohc_comp = []
            for yr in range(ohcready.shape[1]):
                if (act_re[ens,yr] == 1):
                    ohc_comp.append(ohcready[ens,yr,:,:])
            ohc_allenscomp.append(ohc_comp)
    else:
        print(ValueError('SOMETHING IS WRONG WITH ACCURACY COMPOSITES!'))
        sys.exit()
        
    ### Composite hiatuses for 8 ensembles
    meanOHCens = np.empty((len(testindices),lats.shape[0],lons.shape[0]))
    for i in range(len(ohc_allenscomp)):
        if len(ohc_allenscomp) > 0:
            meanOHCens[i,:,:] = np.nanmean(np.asarray(ohc_allenscomp[i]),axis=0)
        else:
            meanOHCens[i,:,:] = np.nan
            
    ### Composite across all ensembles to get hiatuses
    ohcHIATUS[vvv,:,:] = np.nanmean(meanOHCens,axis=0)

### Read in LRP
datalrp = Dataset(directorydata + 'LRP_comp/LRPMap_comp_' + typemodel + '_' + savename + '.nc')
lrp = datalrp.variables['LRP'][:]
datalrp.close()

###############################################################################
###############################################################################
### Plot subplot of observations
letters = ["a","b","c","d","e","f","g","h","i","j","k","l","m","n"]
if rm_ensemble_mean == False:
    limit = np.arange(-1.5,1.51,0.02)
    barlim = np.round(np.arange(-1.5,1.6,0.5),2)
elif rm_ensemble_mean == True:
    limit = np.arange(-1.5,1.6,0.02)
    barlim = np.round(np.arange(-1.5,1.6,0.5),2)
cmap = cmocean.cm.balance
label = r'\textbf{[ HIATUS COMPOSITE ]}'

fig = plt.figure(figsize=(4,8))
###############################################################################
for ppp in range(ohcHIATUS.shape[0]):
    ax1 = plt.subplot(ohcHIATUS.shape[0],1,ppp+1)
    m = Basemap(projection='robin',lon_0=-180,resolution='l',area_thresh=10000)
    m.drawcoastlines(color='darkgrey',linewidth=0.27)
        
    ### Variable
    varn = ohcHIATUS[ppp]
    if ppp == 0:
        lons = np.where(lons >180,lons-360,lons)
        x, y = np.meshgrid(lons,lats)
       
    circle = m.drawmapboundary(fill_color='dimgrey',color='dimgray',
                      linewidth=0.7)
    circle.set_clip_on(False)
    
    cs1 = m.contourf(x,y,varn,limit,extend='both',latlon=True)
    if ppp == 1:
        csc = m.contour(x,y,lrp,np.arange(0.15,0.16,0.1),linestyles='-',latlon=True,
                        colors='gold',linewidths=1)
    
    cs1.set_cmap(cmap) 
    m.fillcontinents(color='dimgrey',lake_color='dimgrey')
            
    ax1.annotate(r'\textbf{[%s]}' % letters[ppp],xy=(0,0),xytext=(0.98,0.89),
                  textcoords='axes fraction',color='k',fontsize=10,
                  rotation=0,ha='center',va='center')
    ax1.annotate(r'\textbf{%s}' % vari_predict[ppp],xy=(0,0),xytext=(0.,0.88),
                  textcoords='axes fraction',color='dimgrey',fontsize=18,
                  rotation=45,ha='center',va='center')
    
###############################################################################
cbar_ax1 = fig.add_axes([0.38,0.05,0.3,0.02])                
cbar1 = fig.colorbar(cs1,cax=cbar_ax1,orientation='horizontal',
                    extend='both',extendfrac=0.07,drawedges=False)
cbar1.set_label(label,fontsize=6,color='dimgrey',labelpad=1.4)  
cbar1.set_ticks(barlim)
cbar1.set_ticklabels(list(map(str,barlim)))
cbar1.ax.tick_params(axis='x', size=.01,labelsize=4)
cbar1.outline.set_edgecolor('dimgrey')

plt.tight_layout()
plt.subplots_adjust(bottom=0.08,wspace=0.01)
if rm_ensemble_mean == True:
    plt.savefig(directoryfigure + 'RawCompositesHiatus_OHClevels_v2_AccH-%s_AccR-%s_rmENSEMBLEmean.png' % (accurate,accurate),dpi=300)
else:
    plt.savefig(directoryfigure + 'RawCompositesHiatus_OHClevels_v2_AccH-%s_AccR-%s.png' % (accurate,accurate),dpi=300)