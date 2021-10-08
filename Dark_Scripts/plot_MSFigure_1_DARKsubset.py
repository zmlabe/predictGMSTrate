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

# for hi in range(yearstrend_mh.squeeze().shape[1]):
#     plt.plot(yearstrend_mh.squeeze()[exampleens,hi],linetrend_mh.squeeze()[exampleens,hi,0]*yearstrend_mh.squeeze()[0,hi]+linetrend_mh.squeeze()[exampleens,hi,1],
#               color='w',linewidth=0.5,zorder=5,clip_on=False)
   
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
    
plt.text(101,SLOPEthreshh_o,r'\textbf{ERA5}',color='gold',fontsize=9,
          va='center')
plt.text(101,(SLOPEthreshh_m*diff_o)[-1],r'\textbf{Threshold}',color='r',fontsize=9,
          va='center')
plt.text(101,timetrend_m[exampleens,-1],r'\textbf{Ensemble \#%s}' % (actualnumberENS),color='lightseagreen',fontsize=9,
          va='center')
plt.text(101,np.nanmean(timetrend_m,axis=0)[-1],r'\textbf{Mean}',color='w',fontsize=9,
          va='center')

plt.tight_layout()
    
plt.savefig(directoryfigure + 'Figure_1_DARK_sub-6.png',dpi=600)