"""
Make figure of LRP maps for manuscript

Author     : Zachary M. Labe
Date       : 7 October 2021
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
import cmasher as cmr
import calc_Utilities as UT
import calc_dataFunctions as df
import calc_Stats as dSS
from sklearn.metrics import accuracy_score

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
    yearsall = np.arange(1990,2090+1,1)
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
    yearsall = np.arange(1990,2090+1,1)
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
directoryfigure = '/Users/zlabe/Documents/Projects/predictGMSTrate/Dark_Figures/'

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
### Read in OHC
data = Dataset(directorydata + 'OHC_comp/OHCcomp_' + 'hiatus_True' + '.nc')
ohc_hiatus = data.variables['OHC100'][:]
data.close()

data = Dataset(directorydata + 'OHC_comp/OHCcomp_' + 'climatechange_True' + '.nc')
ohc_cc = data.variables['OHC100'][:]
data.close()

##############################################################################
##############################################################################
##############################################################################   
### Composite hiatues based on accuracy or not
def LRPcomp(accurateH,accurateR):
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
    
    return lrp_hiatus,lrp_regular

correct_hia,correct_reg = LRPcomp(True,True)
wrong_hia,wrong_reg = LRPcomp('WRONG','WRONG')

##############################################################################
##############################################################################
##############################################################################   
### Average across hiatus and regular period
hiatus_correct = np.nanmean(correct_hia,axis=0)
hiatus_wrong = np.nanmean(wrong_hia,axis=0)
regular_correct = np.nanmean(correct_reg,axis=0)
regular_wrong = np.nanmean(wrong_reg,axis=0)

###############################################################################
###############################################################################
###############################################################################
### Plot subplot of observations
letters = ["a","b","c","d","e","f","g","h","i","j","k","l","m","n"]
limit = np.arange(0,0.81,0.005)
barlim = np.round(np.arange(0,0.81,0.1),2)
cmap = cm.cubehelix2_16.mpl_colormap
label = r'\textbf{LRP [Relevance]}'
limitd = np.arange(-1.5,1.6,0.02)
barlimd = np.round(np.arange(-1.5,1.6,0.5),2)
cmapd = cmocean.cm.balance
labeld = r'\textbf{OHC100 [Normalized]}'

fig = plt.figure(figsize=(10,8))
###############################################################################
ax1 = plt.subplot(221)
m = Basemap(projection='robin',lon_0=-180,resolution='l',area_thresh=10000)
m.drawcoastlines(color='w',linewidth=0.15)
    
### Variable
hiatus_correctz = hiatus_correct/np.max(hiatus_correct)

lon = np.where(lon >180,lon-360,lon)
x, y = np.meshgrid(lon,lat)
   
circle = m.drawmapboundary(fill_color='darkgrey',color='dimgray',
                  linewidth=0.7)
circle.set_clip_on(False)

cs1 = m.contourf(x,y,hiatus_correctz ,limit,extend='max',latlon=True)
cs1.set_cmap(cmap) 

### Box 1
la1 = 25
la2 = 45
lo1 = 140
lo2 = 180+(180-145)
lonsslice = np.linspace(lo1,lo2,lo2-lo1+1)
latsslice = np.ones(len(lonsslice))*la2
m.plot(lonsslice, latsslice, color='aqua', linewidth=1.5, latlon=True,zorder=4)
latsslice = np.ones(len(lonsslice))*la1
m.plot(lonsslice, latsslice, color='aqua', linewidth=1.5, latlon=True,zorder=4)
m.drawgreatcircle(lo1, la1, lo1, la2,linewidth=1.5,color='aqua',zorder=4)
m.drawgreatcircle(lo2, la2, lo2, la1,linewidth=1.5,color='aqua',zorder=4)

### Box 2
la1 = -10
la2 = 10
lo1 = 170
lo2 = 180+(180-90)
lonsslice = np.linspace(lo1,lo2,lo2-lo1+1)
latsslice = np.ones(len(lonsslice))*la2
m.plot(lonsslice, latsslice, color='aqua', linewidth=1.5, latlon=True,zorder=4)
latsslice = np.ones(len(lonsslice))*la1
m.plot(lonsslice, latsslice, color='aqua', linewidth=1.5, latlon=True,zorder=4)
m.drawgreatcircle(lo1, la1, lo1, la2,linewidth=1.5,color='aqua',zorder=4)
m.drawgreatcircle(lo2, la2, lo2, la1,linewidth=1.5,color='aqua',zorder=4)

### Box 3
la1 = -50
la2 = -15
lo1 = 150
lo2 = 180+(180-160)
lonsslice = np.linspace(lo1,lo2,lo2-lo1+1)
latsslice = np.ones(len(lonsslice))*la2
m.plot(lonsslice, latsslice, color='aqua', linewidth=1.5, latlon=True,zorder=4)
latsslice = np.ones(len(lonsslice))*la1
m.plot(lonsslice, latsslice, color='aqua', linewidth=1.5, latlon=True,zorder=4)
m.drawgreatcircle(lo1, la1, lo1, la2,linewidth=1.5,color='aqua',zorder=4)
m.drawgreatcircle(lo2, la2, lo2, la1,linewidth=1.5,color='aqua',zorder=4)
m.fillcontinents(color='k',lake_color='k')

plt.title(r'\textbf{CORRECT SLOWDOWN PREDICTIONS}',fontsize=13,color='w')     
ax1.annotate(r'\textbf{[%s]}' % letters[0],xy=(0,0),xytext=(0.98,0.84),
              textcoords='axes fraction',color='k',fontsize=9,
              rotation=0,ha='center',va='center')

###############################################################################
###############################################################################
###############################################################################
ax2 = plt.subplot(223)
m = Basemap(projection='robin',lon_0=-180,resolution='l',area_thresh=10000)
m.drawcoastlines(color='w',linewidth=0.15)

### Variable
regular_correctz = regular_correct/np.max(regular_correct)

circle = m.drawmapboundary(fill_color='k',color='k',
                  linewidth=0.7)
circle.set_clip_on(False)

cs2 = m.contourf(x,y,regular_correctz,limit,extend='max',latlon=True)
cs2.set_cmap(cmap) 

### Box 1
la1 = 25
la2 = 45
lo1 = 140
lo2 = 180+(180-145)
lonsslice = np.linspace(lo1,lo2,lo2-lo1+1)
latsslice = np.ones(len(lonsslice))*la2
m.plot(lonsslice, latsslice, color='aqua', linewidth=1.5, latlon=True,zorder=4)
latsslice = np.ones(len(lonsslice))*la1
m.plot(lonsslice, latsslice, color='aqua', linewidth=1.5, latlon=True,zorder=4)
m.drawgreatcircle(lo1, la1, lo1, la2,linewidth=1.5,color='aqua',zorder=4)
m.drawgreatcircle(lo2, la2, lo2, la1,linewidth=1.5,color='aqua',zorder=4)

### Box 2
la1 = -10
la2 = 10
lo1 = 170
lo2 = 180+(180-90)
lonsslice = np.linspace(lo1,lo2,lo2-lo1+1)
latsslice = np.ones(len(lonsslice))*la2
m.plot(lonsslice, latsslice, color='aqua', linewidth=1.5, latlon=True,zorder=4)
latsslice = np.ones(len(lonsslice))*la1
m.plot(lonsslice, latsslice, color='aqua', linewidth=1.5, latlon=True,zorder=4)
m.drawgreatcircle(lo1, la1, lo1, la2,linewidth=1.5,color='aqua',zorder=4)
m.drawgreatcircle(lo2, la2, lo2, la1,linewidth=1.5,color='aqua',zorder=4)

### Box 3
la1 = -50
la2 = -15
lo1 = 150
lo2 = 180+(180-160)
lonsslice = np.linspace(lo1,lo2,lo2-lo1+1)
latsslice = np.ones(len(lonsslice))*la2
m.plot(lonsslice, latsslice, color='aqua', linewidth=1.5, latlon=True,zorder=4)
latsslice = np.ones(len(lonsslice))*la1
m.plot(lonsslice, latsslice, color='aqua', linewidth=1.5, latlon=True,zorder=4)
m.drawgreatcircle(lo1, la1, lo1, la2,linewidth=1.5,color='aqua',zorder=4)
m.drawgreatcircle(lo2, la2, lo2, la1,linewidth=1.5,color='aqua',zorder=4)
m.fillcontinents(color='k',lake_color='k')

plt.title(r'\textbf{CORRECT OTHER PREDICTIONS}',fontsize=13,color='w')         
ax2.annotate(r'\textbf{[%s]}' % letters[1],xy=(0,0),xytext=(0.98,0.84),
              textcoords='axes fraction',color='k',fontsize=9,
              rotation=0,ha='center',va='center')

###############################################################################
###############################################################################
###############################################################################
ax2 = plt.subplot(222)
m = Basemap(projection='robin',lon_0=-180,resolution='l',area_thresh=10000)
m.drawcoastlines(color='w',linewidth=0.15)

### Variable
circle = m.drawmapboundary(fill_color='k',color='k',
                  linewidth=0.7)
circle.set_clip_on(False)

cs2 = m.contourf(x,y,ohc_hiatus,limitd,extend='max',latlon=True)
cs2.set_cmap(cmapd) 
csc = m.contour(x,y,hiatus_correctz,np.arange(0.2,1.1,0.2),linestyles='-',latlon=True,
                colors='gold',linewidths=0.7)

m.fillcontinents(color='k',lake_color='k')
plt.title(r'\textbf{SLOWDOWN COMPOSITE}',fontsize=13,color='w')         
ax2.annotate(r'\textbf{[%s]}' % letters[2],xy=(0,0),xytext=(0.98,0.84),
              textcoords='axes fraction',color='k',fontsize=9,
              rotation=0,ha='center',va='center')

###############################################################################
cbar_ax1 = fig.add_axes([0.155,0.1,0.2,0.025])                
cbar1 = fig.colorbar(cs1,cax=cbar_ax1,orientation='horizontal',
                    extend='max',extendfrac=0.07,drawedges=False)
cbar1.set_label(label,fontsize=6,color='darkgrey',labelpad=1.4)  
cbar1.set_ticks(barlim)
cbar1.set_ticklabels(list(map(str,barlim)))
cbar1.ax.tick_params(axis='x', size=.01,labelsize=4)
cbar1.outline.set_edgecolor('darkgrey')

###############################################################################
###############################################################################
###############################################################################
ax2 = plt.subplot(224)
m = Basemap(projection='robin',lon_0=-180,resolution='l',area_thresh=10000)
m.drawcoastlines(color='w',linewidth=0.15)

### Variable
circle = m.drawmapboundary(fill_color='k',color='k',
                  linewidth=0.7)
circle.set_clip_on(False)

cs4 = m.contourf(x,y,ohc_cc,limitd,extend='both',latlon=True)
cs4.set_cmap(cmapd) 

m.fillcontinents(color='k',lake_color='k')
plt.title(r'\textbf{OTHER COMPOSITE}',fontsize=13,color='w')         
ax2.annotate(r'\textbf{[%s]}' % letters[3],xy=(0,0),xytext=(0.98,0.84),
              textcoords='axes fraction',color='k',fontsize=9,
              rotation=0,ha='center',va='center')

###############################################################################
cbar_ax1 = fig.add_axes([0.155,0.1,0.2,0.025])                
cbar1 = fig.colorbar(cs1,cax=cbar_ax1,orientation='horizontal',
                    extend='max',extendfrac=0.07,drawedges=False)
cbar1.set_label(label,fontsize=6,color='darkgrey',labelpad=1.4)  
cbar1.set_ticks(barlim)
cbar1.set_ticklabels(list(map(str,barlim)))
cbar1.ax.tick_params(axis='x', size=.01,labelsize=4)
cbar1.outline.set_edgecolor('darkgrey')

###############################################################################
cbar_axd1 = fig.add_axes([0.65,0.1,0.2,0.025])                
cbard1 = fig.colorbar(cs4,cax=cbar_axd1,orientation='horizontal',
                    extend='both',extendfrac=0.07,drawedges=False)
cbard1.set_label(labeld,fontsize=6,color='darkgrey',labelpad=1.4)  
cbard1.set_ticks(barlimd)
cbard1.set_ticklabels(list(map(str,barlimd)))
cbard1.ax.tick_params(axis='x', size=.01,labelsize=4)
cbard1.outline.set_edgecolor('darkgrey')

plt.tight_layout()
plt.subplots_adjust(hspace=-0.4)
plt.savefig(directoryfigure + 'Figure_3_DARK.png',dpi=300)