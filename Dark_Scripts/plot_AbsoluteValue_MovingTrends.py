"""
Script plots moving trend slopes over time

Author     : Zachary M. Labe
Date       : 18 August 2021
Version    : 1 
"""

### Import packages
import sys
import matplotlib.pyplot as plt
import numpy as np
import calc_Hiatus_v3 as HA
import calc_Utilities as UT
import calc_dataFunctions as df

### Plotting defaults 
plt.rc('text',usetex=True)
plt.rc('font',**{'family':'sans-serif','sans-serif':['Avant Garde']})

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
directoryfigure = '/Users/zlabe/Desktop/GmstTrendPrediction/'
saveData =  monthlychoice + '_' + variq + '_' + reg_name + '_' + dataset_obs
print('*Filename == < %s >' % saveData) 

### Read data
models,lats,lons = read_primary_dataset(variq,dataset,monthlychoice,numOfEns,
                                        lensalso,randomalso,ravelyearsbinary,
                                        ravelbinary,shuffletype,timeper,
                                        lat_bounds,lon_bounds)
obs,lats,lons = read_obs_dataset(variq,dataset_obs,numOfEns,lensalso,randomalso,ravelyearsbinary,ravelbinary,shuffletype,lat_bounds=lat_bounds,lon_bounds=lon_bounds)

### Calculate global mean temperature
lon2,lat2 = np.meshgrid(lons,lats)
modelsm = UT.calc_weightedAve(models,lat2)
obsm = UT.calc_weightedAve(obs,lat2)

trendlength = 10
AGWstart = 1990

SLOPEthreshh_o,diff_o = HA.calc_thresholdOfTrend(obsm,trendlength,yearsobs,AGWstart,'hiatus')
SLOPEthreshh_m,diff_m = HA.calc_thresholdOfTrend(modelsm,trendlength,yearsall,AGWstart,'hiatus')

yearstrend_obsh,linetrend_obsh,indexslopeNegative_obsh,classes_obsh = HA.calc_HiatusAcc(obsm,trendlength,yearsobs,AGWstart,SLOPEthreshh_o,'hiatus',diff_o)
# yearstrend_obsa,linetrend_obsa,indexslopeNegative_obsa,classes_obsa = HA.calc_HiatusAcc(obsm,trendlength,yearsobs,AGWstart,SLOPEthresha,'accel',diffBase)
# classEVENTs_o = HA.combineEvents(classes_obsh,classes_obsa,'obs')

yearstrend_mh,linetrend_mh,indexslopeNegative_mh,classes_mh = HA.calc_HiatusAcc(modelsm,trendlength,yearsall,AGWstart,SLOPEthreshh_m,'hiatus',diff_o)
# yearstrend_ma,linetrend_ma,indexslopeNegative_ma,classes_ma = HA.calc_HiatusAcc(modelsm,trendlength,yearsall,AGWstart,SLOPEthresha,'accel',diffBase)
# classEVENTs_m = HA.combineEvents(classes_mh,classes_ma,'model')

##############################################################################
##############################################################################
##############################################################################
fig = plt.figure()
ax = plt.subplot(111)

### Trends of models and observations
timetrend_o = linetrend_obsh[:,0]
timetrend_m = linetrend_mh[:,:,0]

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
ax.spines['left'].set_color('dimgrey')
ax.spines['bottom'].set_color('dimgrey')
ax.spines['left'].set_linewidth(2)
ax.spines['bottom'].set_linewidth(2)
ax.tick_params('both',length=4,width=2,which='major',color='dimgrey')
ax.tick_params(axis='x',labelsize=7,pad=4)
ax.tick_params(axis='y',labelsize=7,pad=4)

plt.plot(timetrend_m.transpose(),color='dimgrey',linewidth=0.4,alpha=0.5,clip_on=False)
plt.plot(np.nanmean(timetrend_m,axis=0),color='teal',linewidth=3,alpha=1,clip_on=False,
         label=r'\textbf{CESM2-LE}')
plt.plot(timetrend_o,color='maroon',linewidth=4,alpha=1,clip_on=False,
         linestyle='-',label=r'\textbf{ERA5}')

plt.axhline(SLOPEthreshh_o,color='maroon',linewidth=1,linestyle='--',dashes=(1,0.3))
plt.plot(SLOPEthreshh_m*diff_o,color='teal',linewidth=1,linestyle='--',dashes=(1,0.3))

### Plot the hiatus events
classes_mhplot = classes_mh.copy().ravel()
wherehiatus = np.where(classes_mhplot == 1)
classes_mhplot[wherehiatus] = -0.05
classes_mhplot[np.where(classes_mhplot==0)] = np.nan
classes_mhplot = classes_mhplot.reshape(classes_mh.shape)
plt.plot(classes_mhplot.transpose()[:-9,:],color='maroon',marker='o',clip_on=False)
    
plt.ylabel(r'\textbf{10-Year Trends in GMST [$^{\circ}$C/yr]}',fontsize=10,color='dimgrey')
plt.yticks(np.arange(-0.1,0.15,0.05),map(str,np.round(np.arange(-0.1,0.15,0.05),2)))
plt.xticks(np.arange(0,101,10),map(str,np.arange(1990,2101,10)))
plt.xlim([0,100])   
plt.ylim([-0.05,0.1])
plt.subplots_adjust(bottom=0.15)

leg = plt.legend(shadow=False,fontsize=15,loc='upper center',
              bbox_to_anchor=(0.5,1.15),fancybox=True,ncol=4,frameon=False,
              handlelength=0,handletextpad=0)
for line,text in zip(leg.get_lines(), leg.get_texts()):
    text.set_color(line.get_color())

plt.savefig(directoryfigure + 'AbsoluteValue_MovingTrends.png',
            dpi=600)