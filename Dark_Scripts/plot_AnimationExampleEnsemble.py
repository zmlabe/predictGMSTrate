"""
Plot for making an animation showing an example ensemble member of OHC/T2M

Author     : Zachary M. Labe
Date       : 26 October 2021
Version    : 1
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

### Plotting defaults 
plt.rc('text',usetex=True)
plt.rc('font',**{'family':'sans-serif','sans-serif':['Avant Garde']}) 
plt.rc('savefig',facecolor='black')
plt.rc('axes',edgecolor='k')
plt.rc('xtick',color='k')
plt.rc('ytick',color='k')
plt.rc('axes',labelcolor='k')
plt.rc('axes',facecolor='black')

### Parameters
years = np.arange(1979,2099+1)

### Paths
directoryfigure = '/Users/zlabe/Documents/Projects/predictGMSTrate/Dark_Figures/Animation/'

### Read in data
t2m = np.load(directoryfigure + 'T2M_AnimationEnsemble0.npz')['t2m'][:]
ohc = np.load(directoryfigure + 'OHC100_AnimationEnsemble0.npz')['ohc'][:]

data = Dataset('/Users/zlabe/Data/CESM2-LE/monthly/T2M/T2M_1_1850-2100.nc')
lats = data.variables['latitude'][2:-2]
lons = data.variables['longitude'][:]
data.close()

### Calculate trend periods
trendlength = 10
yearsnew = years
yearstrend = np.empty((len(yearsnew)-trendlength+1,trendlength))
datatrend = np.empty((len(yearsnew)-trendlength+1,trendlength))
for hi in range(len(yearsnew)-(trendlength-1)):
    yearstrend[hi,:] = np.arange(yearsnew[hi],yearsnew[hi]+trendlength,1)
    datatrend[hi,:] = t2m[hi:hi+trendlength]

### Calculate trend lines    
linetrend = np.empty((len(yearsnew)-trendlength+1,2))
for hi in range(len(yearsnew)-trendlength+1):         
    linetrend[hi,:] = np.polyfit(yearstrend[hi],datatrend[hi],1)

##############################################################################
##############################################################################
##############################################################################
for i in range(t2m.shape[0]):
    fig = plt.figure(figsize=(8,6))
    ax = plt.subplot(111)
    
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
    ax.spines['left'].set_color('k')
    ax.spines['bottom'].set_color('k')
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    ax.tick_params('both',length=4,width=2,which='major',color='k')
    ax.tick_params(axis='x',labelsize=7,pad=4)
    ax.tick_params(axis='y',labelsize=7,pad=4)
    
    plt.plot(years[:i+1],t2m[:i+1],color='deepskyblue',linewidth=4,alpha=1,clip_on=True)
    # plt.plot(years,t2m,color='deepskyblue',linewidth=4,alpha=1,clip_on=True)
    plt.scatter(years[i],t2m[i],color='crimson',clip_on=True,marker='o',
             s=50,zorder=15)

    if i < 110:
        plt.plot(yearstrend[i,:],linetrend[i,0]*yearstrend[i,:]+linetrend[i,1],
                  color='crimson',linewidth=2.5,zorder=5,clip_on=False)
    
    plt.xlim([1975,2090])   
    plt.ylim([-1,5])
    
    ###############################################################################           
    a = plt.axes([.07, .5, .6, .4])     
    var = ohc[i]
    
    m = Basemap(projection='robin',lon_0=-180,resolution='l',area_thresh=10000)
    m.drawcoastlines(color='darkgrey',linewidth=0.27)
    lons = np.where(lons >180,lons-360,lons)
    x, y = np.meshgrid(lons,lats)
       
    circle = m.drawmapboundary(fill_color='k',color='k',
                      linewidth=0.7)
    circle.set_clip_on(False)
    
    limit = np.arange(-3.5,3.51,0.01)
    cs1 = m.contourf(x,y,var,limit,extend='both',latlon=True)
    cs1.set_cmap(cmocean.cm.balance) 
    m.fillcontinents(color='k',lake_color='k')
    
    plt.tight_layout()
    if i < 10:        
        plt.savefig(directoryfigure + 'Animation_00%s.png' % i,dpi=300)
    elif i < 100:        
        plt.savefig(directoryfigure + 'Animation_0%s.png' % i,dpi=300)
    else:
        plt.savefig(directoryfigure + 'Animation_%s.png' % i,dpi=300)