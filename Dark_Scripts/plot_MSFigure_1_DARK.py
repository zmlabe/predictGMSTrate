"""
Script plots example of hiatus for figure 1

Author     : Zachary M. Labe
Date       : 6 October 2021
Version    : 2
"""

### Import packages
import sys
import matplotlib.pyplot as plt
import numpy as np
import calc_Hiatus_v4 as HA
import calc_Utilities as UT
import calc_dataFunctions as df
import calc_Stats as dSS

### Plotting defaults 
plt.rc('text',usetex=True)
plt.rc('font',**{'family':'sans-serif','sans-serif':['Avant Garde']}) 
plt.rc('savefig',facecolor='black')
plt.rc('axes',edgecolor='darkgrey')
plt.rc('xtick',color='darkgrey')
plt.rc('ytick',color='darkgrey')
plt.rc('axes',labelcolor='darkgrey')
plt.rc('axes',facecolor='black')

###############################################################################
###############################################################################
###############################################################################
### Data preliminaries 
modelGCMs = ['CESM2LE']
dataset_obs = 'ERA5'
allDataLabels = modelGCMs
letters = ["a","b","c","d","e","f","g","h","i","j","k","l","m"]
datasetsingle = ['CESM2le']
monthlychoiceq = ['annual']
variables = ['T2M']
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
baseline = np.arange(1981,2010+1,1)
###############################################################################
###############################################################################
window = 0
yearsall = np.arange(1979+window,2099+1,1)
yearsobs = np.arange(1979+window,2020+1,1)
###############################################################################
###############################################################################
numOfEns = 40
lentime = len(yearsall)
###############################################################################
###############################################################################
dataset = datasetsingle[0]
lat_bounds,lon_bounds = UT.regions(reg_name)
###############################################################################
###############################################################################
ravelyearsbinary = False
ravelbinary = False
lensalso = True
###############################################################################
###############################################################################
###############################################################################
###############################################################################
### Read in model and observational/reanalysis data
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
vv = 0
mo = 0
variq = variables[vv]
monthlychoice = monthlychoiceq[mo]
directoryfigure = '/Users/zlabe/Documents/Projects/predictGMSTrate/Dark_Figures/'
saveData =  monthlychoice + '_' + variq + '_' + reg_name + '_' + dataset_obs
print('*Filename == < %s >' % saveData) 

### Read data
models,lats,lons = read_primary_dataset(variq,dataset,monthlychoice,numOfEns,
                                        lensalso,randomalso,ravelyearsbinary,
                                        ravelbinary,shuffletype,timeper,
                                        lat_bounds,lon_bounds)
obs,lats,lons = read_obs_dataset(variq,dataset_obs,numOfEns,lensalso,randomalso,ravelyearsbinary,ravelbinary,shuffletype,lat_bounds=lat_bounds,lon_bounds=lon_bounds)

models,obs = dSS.calculate_anomalies(models,obs,lats,lons,baseline,yearsall,yearsobs)

### Calculate global mean temperature
lon2,lat2 = np.meshgrid(lons,lats)
modelsm = UT.calc_weightedAve(models,lat2)
obsm = UT.calc_weightedAve(obs,lat2)
meaens = np.nanmean(modelsm[:,:].squeeze(),axis=0)
maxens = np.nanmax(modelsm[:,:].squeeze(),axis=0)
minens = np.nanmin(modelsm[:,:].squeeze(),axis=0)
spread = maxens - minens

trendlength = 10
AGWstart = 1990

SLOPEthreshh_o,diff_o = HA.calc_thresholdOfTrend(obsm,trendlength,yearsobs,AGWstart,'hiatus')
SLOPEthreshh_m,diff_m = HA.calc_thresholdOfTrend(modelsm,trendlength,yearsall,AGWstart,'hiatus')
yearstrend_obsh,linetrend_obsh,indexslopeNegative_obsh,classes_obsh = HA.calc_HiatusAcc(obsm,trendlength,yearsobs,AGWstart,SLOPEthreshh_o,'hiatus',diff_o)
yearstrend_mh,linetrend_mh,indexslopeNegative_mh,classes_mh = HA.calc_HiatusAcc(modelsm,trendlength,yearsall,AGWstart,SLOPEthreshh_m,'hiatus',diff_o)

##############################################################################
##############################################################################
##############################################################################
fig = plt.figure(figsize=(6,8))
ax = plt.subplot(211)

### Trends of models and observations
timetrend_o = linetrend_obsh[:,0]
timetrend_m = linetrend_mh[:,:,0]

### Pick example ensemble member 
exampleens = 2
actualnumberENS = ens[exampleens]

def adjust_spines(ax, spines):
    for loc, spine in ax.spines.items():
        if loc in spines:
            spine.set_position(('outward', 5))
        else:
            spine.set_color('none')  
    if 'left' in spines:
        ax.yaxis.set_ticks_position('left')
    else:
        ax.yaxis.set_ticks([])

    if 'bottom' in spines:
        ax.xaxis.set_ticks_position('bottom')
    else:
        ax.xaxis.set_ticks([])
        
adjust_spines(ax, ['left', 'bottom'])
ax.spines['top'].set_color('none')
ax.spines['right'].set_color('none')
ax.spines['left'].set_color('darkgrey')
ax.spines['bottom'].set_color('darkgrey')
ax.spines['left'].set_linewidth(2)
ax.spines['bottom'].set_linewidth(2)
ax.tick_params('both',length=4,width=2,which='major',color='darkgrey')
ax.tick_params(axis='x',labelsize=7,pad=4)
ax.tick_params(axis='y',labelsize=7,pad=4)

ax.fill_between(yearsall[11:112],minens[11:112],
                maxens[11:112],facecolor='w',
                alpha=0.35,zorder=1,clip_on=False)

plt.plot(yearsall,modelsm.squeeze()[exampleens],color='lightseagreen',linewidth=3,
         alpha=1)
plt.plot(yearsobs,obsm,color='gold',alpha=1,linewidth=2,linestyle='-')
for hi in range(yearstrend_mh.squeeze().shape[1]):
    if linetrend_mh.squeeze()[exampleens,hi,0] <= SLOPEthreshh_m[hi]*diff_o:
        plt.plot(yearstrend_mh.squeeze()[exampleens,hi],linetrend_mh.squeeze()[exampleens,hi,0]*yearstrend_mh.squeeze()[0,hi]+linetrend_mh.squeeze()[exampleens,hi,1],
                  color='r',linewidth=1.2,zorder=5,clip_on=False)
   
yearhiatusq = np.where((yearsall>=AGWstart) & (yearsall<=2090))[0]
yearsallhiatus = yearsall[yearhiatusq]
classes_mhplot = classes_mh.copy().ravel()
wherehiatus = np.where(classes_mhplot == 1)
classes_mhplot[wherehiatus] = -0.5
classes_mhplot[np.where(classes_mhplot==0)] = np.nan
classes_mhplot = classes_mhplot.reshape(classes_mh.shape)
plt.plot(yearsallhiatus,classes_mhplot[exampleens,:],color='r',marker='o',clip_on=False,
         markersize=10,label=r'\textbf{SLOWDOWN EVENT')

plt.text(2091,modelsm.squeeze()[exampleens,-10],r'\textbf{Ensemble \#%s}' % (actualnumberENS),
         color='lightseagreen',fontsize=9,va='center')

leg = plt.legend(shadow=False,fontsize=20,loc='upper center',
              bbox_to_anchor=(0.5,1.05),fancybox=True,ncol=1,frameon=False,
              handlelength=0)
for line,text in zip(leg.get_lines(), leg.get_texts()):
    text.set_color(line.get_color())

plt.ylabel(r'\textbf{GMST Anomaly [$^{\circ}$C]}',fontsize=10,color='w',labelpad=8)
plt.yticks(np.arange(-5,21,0.5),map(str,np.round(np.arange(-5,21,0.5),2)),size=9)
plt.xticks(np.arange(1850,2100+1,10),map(str,np.arange(1850,2100+1,10)),size=9)
plt.xlim([1990,2090])   
plt.ylim([-0.5,4])

plt.text(2030,3.2,r'\textbf{CESM2}',fontsize=20,color='darkgrey')
plt.text(2033.3,2.8,r'\textbf{ERA5}',fontsize=20,color='gold')
plt.text(1990,4.0,r'\textbf{[a]}',color='k',fontsize=10)

##############################################################################
##############################################################################
##############################################################################
ax = plt.subplot(212)
adjust_spines(ax, ['left', 'bottom'])
ax.spines['top'].set_color('none')
ax.spines['right'].set_color('none')
ax.spines['left'].set_color('darkgrey')
ax.spines['bottom'].set_color('darkgrey')
ax.spines['left'].set_linewidth(2)
ax.spines['bottom'].set_linewidth(2)
ax.tick_params('both',length=4,width=2,which='major',color='darkgrey')
ax.tick_params(axis='x',labelsize=7,pad=4)
ax.tick_params(axis='y',labelsize=7,pad=4)

plt.plot(timetrend_m.transpose(),color='w',linewidth=0.4,alpha=0.35,clip_on=False)
plt.plot(np.nanmean(timetrend_m,axis=0),color='w',linewidth=1.5,alpha=1,clip_on=False,
         label=r'\textbf{CESM2-LE MEAN}')
plt.plot(timetrend_m[exampleens,:],color='lightseagreen',linewidth=3,alpha=1,clip_on=False)
plt.plot(timetrend_o,color='gold',linewidth=4,alpha=1,clip_on=False,
         linestyle='-',label=r'\textbf{ERA5}')

plt.axhline(SLOPEthreshh_o,color='gold',linewidth=1,linestyle='--',dashes=(3,3))
plt.plot(SLOPEthreshh_m*diff_o,color='r',linewidth=1,linestyle='--',dashes=(3,3))

### Plot the hiatus events
classes_mhplot = classes_mh.copy().ravel()
wherehiatus = np.where(classes_mhplot == 1)
classes_mhplot[wherehiatus] = -0.05
classes_mhplot[np.where(classes_mhplot==0)] = np.nan
classes_mhplot = classes_mhplot.reshape(classes_mh.shape)
plt.plot(classes_mhplot[exampleens,:],color='r',marker='o',clip_on=False,
         markersize=10)
    
plt.ylabel(r'\textbf{10-Year Trends in GMST [$^{\circ}$C/yr]}',fontsize=10,color='w')
plt.yticks(np.arange(-0.1,0.15,0.05),map(str,np.round(np.arange(-0.1,0.15,0.05),2)),size=9)
plt.xticks(np.arange(0,101,10),map(str,np.arange(1990,2101,10)),size=9)
plt.xlim([0,100])   
plt.ylim([-0.05,0.1])
plt.subplots_adjust(bottom=0.15)
    
plt.text(0,0.1,r'\textbf{[b]}',color='k',fontsize=10)
plt.text(101,SLOPEthreshh_o,r'\textbf{ERA5 Threshold}',color='gold',fontsize=9,
         va='center')
plt.text(101,(SLOPEthreshh_m*diff_o)[-1],r'\textbf{CESM2-LE Threshold}',color='r',fontsize=9,
         va='center')
plt.text(101,timetrend_m[exampleens,-1],r'\textbf{Ensemble \#%s}' % (actualnumberENS),color='lightseagreen',fontsize=9,
         va='center')
plt.text(101,np.nanmean(timetrend_m,axis=0)[-1],r'\textbf{Ensemble Mean}',color='w',fontsize=9,
         va='center')

plt.tight_layout()

##############################################################################
##############################################################################
##############################################################################
##############################################################################
##############################################################################
##############################################################################
##############################################################################
##############################################################################
##############################################################################

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
    yearsall = np.arange(1990,2090+1,1)
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

###############################################################################
###############################################################################
###############################################################################
### Read in data for actual hiatuses in training
trainindices = np.genfromtxt(directorydata + 'trainingEnsIndices_' + savename + '.txt')
actual_train = np.genfromtxt(directorydata + 'trainingTrueLabels_' + savename + '.txt')
act_retrain = np.swapaxes(actual_train.reshape(trainindices.shape[0],1,yearsall.shape[0]),0,1).squeeze()

###############################################################################
###############################################################################
###############################################################################
### Read in data for actual hiatuses in testing
testindices = np.genfromtxt(directorydata + 'testingEnsIndices_' + savename + '.txt')
actual_test = np.genfromtxt(directorydata + 'testingTrueLabels_' + savename + '.txt')
act_retest = np.swapaxes(actual_test.reshape(testindices.shape[0],1,yearsall.shape[0]),0,1).squeeze()

###############################################################################
###############################################################################
###############################################################################
### Read in data for actual hiatuses in validation
valindices = np.genfromtxt(directorydata + 'validationEnsIndices_' + savename + '.txt')
actual_val = np.genfromtxt(directorydata + 'validationTrueLabels_' + savename + '.txt')
act_reval = np.swapaxes(actual_val.reshape(valindices.shape[0],1,yearsall.shape[0]),0,1).squeeze()

### Count number of hiatus per ensemble member for time period
yrq = np.where((yearsall == 2040))[0][0]
traincount_pre = np.count_nonzero(act_retrain[:,:yrq]==1,axis=1)
testcount_pre  = np.count_nonzero(act_retest[:,:yrq]==1,axis=1)
valcount_pre  = np.count_nonzero(act_reval[:,:yrq]==1,axis=1)

traincount_pos = np.count_nonzero(act_retrain[:,yrq:]==1,axis=1)
testcount_pos  = np.count_nonzero(act_retest[:,yrq:]==1,axis=1)
valcount_pos = np.count_nonzero(act_reval[:,yrq:]==1,axis=1)

### Concatenate
totalcounts_pre = np.concatenate((traincount_pre,testcount_pre,valcount_pre))
totalcounts_pos = np.concatenate((traincount_pos,testcount_pos,valcount_pos))

ax = plt.axes([.6,.6,.35,.12])
        
adjust_spines(ax, ['left','bottom'])            
ax.spines['top'].set_color('none')
ax.spines['right'].set_color('none') 
ax.spines['bottom'].set_color('darkgrey')
ax.spines['left'].set_color('darkgrey')
ax.spines['bottom'].set_linewidth(1)
ax.spines['left'].set_linewidth(1) 
ax.tick_params('both',length=3.5,width=1,which='major',color='darkgrey')  
ax.yaxis.grid(zorder=1,color='darkgrey',alpha=0.35)

### Plot histograms
weights_count_pre = np.ones_like(totalcounts_pre)/len(totalcounts_pre)
n_count_pre, bins_count_pre, patches_count_pre = plt.hist(totalcounts_pre,bins=np.arange(-1,21,1)-0.5,
                                        density=False,alpha=1,
                                        label=r'\textbf{1990-2039}',
                                        weights=weights_count_pre,zorder=3)
for i in range(len(patches_count_pre)):
    patches_count_pre[i].set_facecolor('teal')
    patches_count_pre[i].set_edgecolor('white')
    patches_count_pre[i].set_linewidth(0.6)
    
weights_count_pos = np.ones_like(totalcounts_pos)/len(totalcounts_pos)
n_count_pos, bins_count_pos, patches_count_pos = plt.hist(totalcounts_pos,bins=np.arange(-1,21,1)-0.5,
                                        density=False,alpha=0.6,
                                        label=r'\textbf{2040-2090}',
                                        weights=weights_count_pos,zorder=3)
for i in range(len(patches_count_pre)):
    patches_count_pos[i].set_facecolor('salmon')
    patches_count_pos[i].set_edgecolor('white')
    patches_count_pos[i].set_linewidth(0.6)

leg = plt.legend(shadow=False,fontsize=6,loc='upper center',
                  bbox_to_anchor=(0.5,1.21),fancybox=True,ncol=2,frameon=False,
                  handlelength=3,handletextpad=1)
for text in leg.get_texts():
    text.set_color('darkgrey')

plt.text(-1,0.42,r'\textbf{[c]}',color='k',fontsize=10)

plt.ylabel(r'\textbf{Frequency}',fontsize=7,color='darkgrey')
plt.xlabel(r'\textbf{Number of slowdown events per member}',fontsize=7,color='darkgrey')
plt.yticks(np.arange(0,1.1,0.1),map(str,np.round(np.arange(0,1.1,0.1),2)),size=6)
plt.xticks(np.arange(0,21,2),map(str,np.round(np.arange(0,21,2),2)),size=6)
plt.xlim([-0.5,20])   
plt.ylim([0,0.4])
    
plt.savefig(directoryfigure + 'Figure_1_DARK.png',dpi=600)