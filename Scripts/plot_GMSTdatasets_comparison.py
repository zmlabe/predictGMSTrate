"""
Script plots a comparison of GMST data sets for their trends

Author     : Zachary M. Labe
Date       : 10 January 2022
Version    : 2
Revision   : 1
"""

### Import packages
import sys
import matplotlib.pyplot as plt
import numpy as np
import calc_Hiatus_v4 as HA
import calc_Utilities as UT
import calc_dataFunctions as df
import calc_Stats as dSS
import cmocean
import cmasher as cmr

### Plotting defaults 
plt.rc('text',usetex=True)
plt.rc('font',**{'family':'sans-serif','sans-serif':['Avant Garde']})

###############################################################################
###############################################################################
###############################################################################
### Data preliminaries 
dataset_obs = ['ERA5','BEST','GISTEMP','HadCRUT','NCEP2']
obslabels = ['ERA5','BEST','GISTEMPv4','HadCRUT5','NCEP2']
letters = ["a","b","c","d","e","f","g","h","i","j","k","l","m"]
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
baseline = np.arange(1981,2010+1,1)
###############################################################################
###############################################################################
window = 0
yearsobs = [np.arange(1979+window,2020+1,1),np.arange(1850+window,2020+1,1),
            np.arange(1880+window,2020+1,1),np.arange(1850+window,2020+1,1),
            np.arange(1979+window,2020+1,1)]
###############################################################################
###############################################################################
numOfEns = 40
lentime = len(yearsobs)
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
###############################################################################
###############################################################################
### Read in observational/reanalysis data
def read_obs_dataset(variq,dataset_obs,numOfEns,lensalso,randomalso,ravelyearsbinary,ravelbinary,shuffletype,lat_bounds=lat_bounds,lon_bounds=lon_bounds):
    data_obs,lats_obs,lons_obs = df.readFiles(variq,dataset_obs,monthlychoice,numOfEns,lensalso,randomalso,ravelyearsbinary,ravelbinary,shuffletype,timeper)
    data_obs,lats_obs,lons_obs = df.getRegion(data_obs,lats_obs,lons_obs,lat_bounds,lon_bounds)
    
    print('our OBS dataset: ',dataset_obs,' is shaped',data_obs.shape)
    return data_obs,lats_obs,lons_obs

###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
### Try different data sets    

SLOPEthreshh_o = []
diff_o = []
yearstrend_obsh = []
linetrend_obsh = []
indexslopeNegative_obsh = []
classes_obsh = []
for i in range(len(dataset_obs)):
    ### Call functions
    vv = 0
    mo = 0
    variq = variables[vv]
    monthlychoice = monthlychoiceq[mo]
    directoryfigure = '/Users/zlabe/Desktop/GmstTrendPrediction/MS-Figures_v2/'
    saveData =  monthlychoice + '_' + variq + '_' + reg_name + '_' + dataset_obs[i]
    print('*Filename == < %s >' % saveData) 

    ### Read data
    obs,lats,lons = read_obs_dataset(variq,dataset_obs[i],numOfEns,lensalso,randomalso,ravelyearsbinary,ravelbinary,shuffletype,lat_bounds=lat_bounds,lon_bounds=lon_bounds)
    
    ### Calculate global mean temperature
    lon2,lat2 = np.meshgrid(lons,lats)
    obsm = UT.calc_weightedAve(obs,lat2)
    
    if dataset_obs == 'NCEP2':
        obsm = obsm[:-1] # does not include 2021
    
    trendlength = 10
    AGWstart = 1990
    
    SLOPEthreshh_oq,diff_oq = HA.calc_thresholdOfTrend(obsm,trendlength,yearsobs[i],AGWstart,'hiatus')
    yearstrend_obshq,linetrend_obshq,indexslopeNegative_obshq,classes_obshq = HA.calc_HiatusAcc(obsm,trendlength,yearsobs[i],AGWstart,SLOPEthreshh_oq,'hiatus',diff_oq)

    SLOPEthreshh_o.append(SLOPEthreshh_oq)
    diff_o.append(diff_oq)
    yearstrend_obsh.append(yearstrend_obshq)
    linetrend_obsh.append(linetrend_obshq)
    indexslopeNegative_obsh.append(indexslopeNegative_obshq)
    classes_obsh.append(classes_obshq)

##############################################################################
##############################################################################
##############################################################################
fig = plt.figure()
ax = plt.subplot(111)

### Trends of models and observations
timetrend_o = np.asarray(linetrend_obsh)[:,:,0]

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
        
##############################################################################
##############################################################################
##############################################################################
ax = plt.subplot(111)
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

plt.fill_between(x=np.arange(12,16,1),y1=-0.02,y2=0.06,facecolor='darkgrey',zorder=0,
         alpha=0.3,edgecolor='none',clip_on=False)
plt.axhline(0,color='dimgrey',alpha=1,linestyle='-',linewidth=2,clip_on=False)

color = iter(cmr.infinity(np.linspace(0.00,1,len(timetrend_o))))
for i in range(timetrend_o.shape[0]):
    if i == 0:
        cma = 'k'
        ll = 3
        aa = 1
        plt.plot(timetrend_o[i,:],color=cma,alpha=aa,linewidth=ll,clip_on=False,
             label=r'\textbf{%s}' % obslabels[i],linestyle='--',dashes=(1,0.3))
        plt.axhline(SLOPEthreshh_o[i],color=cma,alpha=aa,linestyle='-',linewidth=0.5,clip_on=False)
    else:
        cma=next(color)
        ll = 2
        aa = 1
        plt.plot(timetrend_o[i,:],color=cma,alpha=aa,linewidth=ll,clip_on=False,
             label=r'\textbf{%s}' % obslabels[i])
        plt.axhline(SLOPEthreshh_o[i],color=cma,alpha=aa,linestyle='-',linewidth=0.5,clip_on=False)
    
leg = plt.legend(shadow=False,fontsize=10,loc='upper center',
            bbox_to_anchor=(0.51, 1.15),fancybox=True,ncol=15,frameon=False,
            handlelength=1,handletextpad=0.5)
for line,text in zip(leg.get_lines(), leg.get_texts()):
    text.set_color(line.get_color())
    
plt.ylabel(r'\textbf{10-Year Trends in GMST [$^{\circ}$C/yr]}',fontsize=10,color='k')
plt.yticks(np.arange(-0.1,0.15,0.02),map(str,np.round(np.arange(-0.1,0.15,0.02),2)),size=9)
plt.xticks(np.arange(0,101,5),map(str,np.arange(1990,2101,5)),size=9)
plt.xlim([0,21])   
plt.ylim([-0.02,0.06])
plt.subplots_adjust(bottom=0.15)
plt.savefig(directoryfigure + 'GMST_TrendsComparison.png',dpi=600)