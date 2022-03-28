"""
LRP composites for the wrong predictions

Author     : Zachary M. Labe
Date       : 28 March 2022
Version    : 2
"""

### Import packages
import sys
import matplotlib.pyplot as plt
import numpy as np
from netCDF4 import Dataset
import scipy.stats as stats
from mpl_toolkits.basemap import Basemap, addcyclic, shiftgrid
import palettable.cubehelix as cm
import cmocean as cmocean
import calc_Utilities as UT
import calc_dataFunctions as df
import calc_Stats as dSS
from sklearn.metrics import accuracy_score

### Plotting defaults 
plt.rc('text',usetex=True)
plt.rc('font',**{'family':'sans-serif','sans-serif':['Avant Garde']})

### Parameters FOR THIS SCRIPT
accurateR = 'WRONG'
accurateH = 'WRONG'

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

### Naming conventions for files
directorymodel = '/Users/zlabe/Documents/Research/GmstTrendPrediction/SavedModels/'
savename = 'ANNv2_'+vari_predict[0]+'_hiatus_' + actFun + '_L2_'+ str(ridgePenalty)+ '_LR_' + str(lr_here)+ '_Batch'+ str(batch_size)+ '_Iters' + str(n_epochs) + '_' + str(len(hidden)) + 'x' + str(hidden[0]) + '_SegSeed' + str(random_segment_seed) + '_NetSeed'+ str(random_network_seed) 
if(rm_ensemble_mean==True):
    savename = savename + '_EnsembleMeanRemoved'  

### Directories to save files
directorydata = '/Users/zlabe/Documents/Research/GmstTrendPrediction/Data/'
directoryfigure = '/Users/zlabe/Desktop/GmstTrendPrediction/ANN_v2/LRP/'

###############################################################################
###############################################################################
###############################################################################
### Read in LRP data for testing and obs
nametypetest = 'Testing'
filenametest = directorydata  + 'LRPMap' + nametypetest + '_' + savename + '.nc'
datatest = Dataset(filenametest,'r')
lat = datatest.variables['lat'][:]
lon = datatest.variables['lon'][:]
lrp_test = datatest.variables['LRP'][:]
datatest.close()

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

##############################################################################
##############################################################################
##############################################################################   
### Composite hiatues based on accuracy or not
lrp_hiatus = []
for i in range(lrp_test.shape[0]):
    if accurateH == False:
        if predict_test[i] == 1:
            lrp_hiatus.append(lrp_test[i])
    elif accurateH == True:
        if (predict_test[i]) == 1 and (actual_test[i] == 1):
            lrp_hiatus.append(lrp_test[i])
    elif accurateH == 'WRONG':
        if (predict_test[i]) == 1 and (actual_test[i] == 0):
            lrp_hiatus.append(lrp_test[i])
    else:
        print(ValueError('WRONG COMPOSITE METRIC!!'))
        sys.exit()
lrp_hiatus = np.asarray(lrp_hiatus)

##############################################################################
##############################################################################
##############################################################################   
### Composite regular period based on accuracy or not
lrp_regular = []
for i in range(lrp_test.shape[0]):
    if accurateR == False:
        if predict_test[i] == 0:
            lrp_regular.append(lrp_test[i])
    elif accurateR == True:
        if (predict_test[i]) == 0 and (actual_test[i] == 0):
            lrp_regular.append(lrp_test[i])
    elif accurateR == 'WRONG':
        if (predict_test[i]) == 0 and (actual_test[i] == 1):
            lrp_regular.append(lrp_test[i])
    else:
        print(ValueError('WRONG COMPOSITE METRIC!!'))
        sys.exit()
lrp_regular  = np.asarray(lrp_regular)

##############################################################################
##############################################################################
##############################################################################   
### Average across hiatus and regular period
hiatuslrpm = np.nanmean(lrp_hiatus,axis=0)
regularlrpm = np.nanmean(lrp_regular,axis=0)

###############################################################################
###############################################################################
###############################################################################
### Plot subplot of observations
letters = ["a","b","c","d","e","f","g","h","i","j","k","l","m","n"]
limit = np.arange(0,0.81,0.005)
barlim = np.round(np.arange(0,0.81,0.1),2)
cmap = cm.cubehelix2_16.mpl_colormap
label = r'\textbf{LRP [Relevancce]}'

fig = plt.figure(figsize=(8,3))
###############################################################################
ax1 = plt.subplot(121)
m = Basemap(projection='robin',lon_0=-180,resolution='l',area_thresh=10000)
m.drawcoastlines(color='darkgrey',linewidth=0.27)

### Variable
circle = m.drawmapboundary(fill_color='dimgrey',color='dimgray',
                  linewidth=0.7)
    
### Variable
var_R = regularlrpm/np.max(regularlrpm)

lon = np.where(lon >180,lon-360,lon)
x, y = np.meshgrid(lon,lat)
   
circle.set_clip_on(False)

cs1 = m.contourf(x,y,var_R ,limit,extend='max',latlon=True)
cs1.set_cmap(cmap) 

# ### Box 1
# la1 = 25
# la2 = 45
# lo1 = 140
# lo2 = 180+(180-145)
# lonsslice = np.linspace(lo1,lo2,lo2-lo1+1)
# latsslice = np.ones(len(lonsslice))*la2
# m.plot(lonsslice, latsslice, color='aqua', linewidth=1.5, latlon=True,zorder=4)
# latsslice = np.ones(len(lonsslice))*la1
# m.plot(lonsslice, latsslice, color='aqua', linewidth=1.5, latlon=True,zorder=4)
# m.drawgreatcircle(lo1, la1, lo1, la2,linewidth=1.5,color='aqua',zorder=4)
# m.drawgreatcircle(lo2, la2, lo2, la1,linewidth=1.5,color='aqua',zorder=4)

# ### Box 2
# la1 = -10
# la2 = 10
# lo1 = 170
# lo2 = 180+(180-90)
# lonsslice = np.linspace(lo1,lo2,lo2-lo1+1)
# latsslice = np.ones(len(lonsslice))*la2
# m.plot(lonsslice, latsslice, color='aqua', linewidth=1.5, latlon=True,zorder=4)
# latsslice = np.ones(len(lonsslice))*la1
# m.plot(lonsslice, latsslice, color='aqua', linewidth=1.5, latlon=True,zorder=4)
# m.drawgreatcircle(lo1, la1, lo1, la2,linewidth=1.5,color='aqua',zorder=4)
# m.drawgreatcircle(lo2, la2, lo2, la1,linewidth=1.5,color='aqua',zorder=4)

# ### Box 3
# la1 = -50
# la2 = -15
# lo1 = 150
# lo2 = 180+(180-160)
# lonsslice = np.linspace(lo1,lo2,lo2-lo1+1)
# latsslice = np.ones(len(lonsslice))*la2
# m.plot(lonsslice, latsslice, color='aqua', linewidth=1.5, latlon=True,zorder=4)
# latsslice = np.ones(len(lonsslice))*la1
# m.plot(lonsslice, latsslice, color='aqua', linewidth=1.5, latlon=True,zorder=4)
# m.drawgreatcircle(lo1, la1, lo1, la2,linewidth=1.5,color='aqua',zorder=4)
# m.drawgreatcircle(lo2, la2, lo2, la1,linewidth=1.5,color='aqua',zorder=4)
# m.fillcontinents(color='dimgrey',lake_color='dimgrey')

m.fillcontinents(color='dimgrey',lake_color='dimgrey')
plt.title(r'\textbf{WRONG \textit{NO} SLOWDOWN PREDICTIONS}',fontsize=13,color='dimgrey')
ax1.annotate(r'\textbf{%s Cases}' % len(lrp_regular),xy=(0,0),xytext=(0.05,0.83),
              textcoords='axes fraction',color='k',fontsize=9,
              rotation=47,ha='center',va='center')           
ax1.annotate(r'\textbf{[%s]}' % letters[0],xy=(0,0),xytext=(0.98,0.84),
              textcoords='axes fraction',color='k',fontsize=9,
              rotation=0,ha='center',va='center')

###############################################################################
ax2 = plt.subplot(122)
m = Basemap(projection='robin',lon_0=-180,resolution='l',area_thresh=10000)
m.drawcoastlines(color='darkgrey',linewidth=0.27)

### Variable
circle = m.drawmapboundary(fill_color='dimgrey',color='dimgray',
                  linewidth=0.7)
### Variable
var_H = hiatuslrpm/np.max(hiatuslrpm)


circle.set_clip_on(False)

cs2 = m.contourf(x,y,var_H,limit,extend='max',latlon=True)
cs2.set_cmap(cmap) 

# ### Box 1
# la1 = 25
# la2 = 45
# lo1 = 140
# lo2 = 180+(180-145)
# lonsslice = np.linspace(lo1,lo2,lo2-lo1+1)
# latsslice = np.ones(len(lonsslice))*la2
# m.plot(lonsslice, latsslice, color='aqua', linewidth=1.5, latlon=True,zorder=4)
# latsslice = np.ones(len(lonsslice))*la1
# m.plot(lonsslice, latsslice, color='aqua', linewidth=1.5, latlon=True,zorder=4)
# m.drawgreatcircle(lo1, la1, lo1, la2,linewidth=1.5,color='aqua',zorder=4)
# m.drawgreatcircle(lo2, la2, lo2, la1,linewidth=1.5,color='aqua',zorder=4)

# ### Box 2
# la1 = -10
# la2 = 10
# lo1 = 170
# lo2 = 180+(180-90)
# lonsslice = np.linspace(lo1,lo2,lo2-lo1+1)
# latsslice = np.ones(len(lonsslice))*la2
# m.plot(lonsslice, latsslice, color='aqua', linewidth=1.5, latlon=True,zorder=4)
# latsslice = np.ones(len(lonsslice))*la1
# m.plot(lonsslice, latsslice, color='aqua', linewidth=1.5, latlon=True,zorder=4)
# m.drawgreatcircle(lo1, la1, lo1, la2,linewidth=1.5,color='aqua',zorder=4)
# m.drawgreatcircle(lo2, la2, lo2, la1,linewidth=1.5,color='aqua',zorder=4)

# ### Box 3
# la1 = -50
# la2 = -15
# lo1 = 150
# lo2 = 180+(180-160)
# lonsslice = np.linspace(lo1,lo2,lo2-lo1+1)
# latsslice = np.ones(len(lonsslice))*la2
# m.plot(lonsslice, latsslice, color='aqua', linewidth=1.5, latlon=True,zorder=4)
# latsslice = np.ones(len(lonsslice))*la1
# m.plot(lonsslice, latsslice, color='aqua', linewidth=1.5, latlon=True,zorder=4)
# m.drawgreatcircle(lo1, la1, lo1, la2,linewidth=1.5,color='aqua',zorder=4)
# m.drawgreatcircle(lo2, la2, lo2, la1,linewidth=1.5,color='aqua',zorder=4)
# m.fillcontinents(color='dimgrey',lake_color='dimgrey')

m.fillcontinents(color='dimgrey',lake_color='dimgrey')
plt.title(r'\textbf{WRONG SLOWDOWN PREDICTIONS}',fontsize=13,color='dimgrey')
ax2.annotate(r'\textbf{%s Cases}' % len(lrp_hiatus),xy=(0,0),xytext=(0.05,0.83),
              textcoords='axes fraction',color='k',fontsize=9,
              rotation=47,ha='center',va='center')           
ax2.annotate(r'\textbf{[%s]}' % letters[1],xy=(0,0),xytext=(0.98,0.84),
              textcoords='axes fraction',color='k',fontsize=9,
              rotation=0,ha='center',va='center')

###############################################################################
cbar_ax1 = fig.add_axes([0.40,0.1,0.2,0.05])                
cbar1 = fig.colorbar(cs1,cax=cbar_ax1,orientation='horizontal',
                    extend='max',extendfrac=0.07,drawedges=False)
cbar1.set_label(label,fontsize=6,color='dimgrey',labelpad=1.4)  
cbar1.set_ticks(barlim)
cbar1.set_ticklabels(list(map(str,barlim)))
cbar1.ax.tick_params(axis='x', size=.01,labelsize=4)
cbar1.outline.set_edgecolor('dimgrey')

plt.tight_layout()
if rm_ensemble_mean == True:
    plt.savefig(directoryfigure + 'LRP_wrongPredictions_SUPP.png',dpi=600)
