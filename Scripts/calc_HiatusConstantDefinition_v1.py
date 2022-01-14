"""
Functions calculate hiatus and acceleration definitions
 
Notes
-----
    Author  : Zachary Labe
    Date    : 22 January 2022
    Version : 1 
    
Usage
-----
    [1] calc_thresholdOfTrend(data,trendlength,years,AGWstart,typeOfTrend)
    [2] calc_HiatusAcc(data,trendlength,years,AGWstart,SLOPEthresh,typeOfTrend)
    [3] combineEvents(hiatus,accel,typeOfData)
"""
def calc_thresholdOfTrend(data,trendlength,years,AGWstart,typeOfTrend): 
    """
    Function calculates threshold for trend analysis of hiatus or acceleration

    Parameters
    ----------
    data : n-d numpy array
        data from selected data set
    trendlength : integer
        length of trend periods to calculate
    years : 1d array
        Original years of input data
    AGWstart : integer
        Start of data to calculate trends over
    typeOfTrend : string
        hiatus or accel
        
    Returns
    -------
    SLOPEthresh : float 
        float of the actual trend for hiatus or acceleration
    obsdiff : float
        difference compared to mean decadal trencds and the hiatus/acceleration

    Usage
    -----
    SLOPEthresh,obsdiff = calc_thresholdOfTrend(data,trendlength,years,AGWstart,typeOfTrend)
    """
    print('\n>>>>>>>>>> Using calc_thresholdOfTrend function!')
    
    ### Import modules
    import numpy as np
    import sys
    
    if data.ndim == 1: 
        ### Pick start of forced climate change
        yrq = np.where(years[:] >= AGWstart)[0]
        data = data[yrq]
        yearsnew = years[yrq]
        print('Years-Trend ---->\n',yearsnew)
        
    ### Calculate trend periods
        yearstrend = np.empty((len(yearsnew)-trendlength+1,trendlength))
        datatrend = np.empty((len(yearsnew)-trendlength+1,trendlength))
        for hi in range(len(yearsnew)-(trendlength-1)):
            yearstrend[hi,:] = np.arange(yearsnew[hi],yearsnew[hi]+trendlength,1)
            datatrend[hi,:] = data[hi:hi+trendlength]

        ### Calculate trend lines    
        linetrend = np.empty((len(yearsnew)-trendlength+1,2))
        for hi in range(len(yearsnew)-trendlength+1):
            linetrend[hi,:] = np.polyfit(yearstrend[hi],datatrend[hi],1)
            
        ### Slopes
        slope = linetrend[:,0]
        stdslope = np.nanstd(slope)
        meantrend = np.nanmean(slope)
        print(meantrend,stdslope,slope)
        
        print('\n**%s** is %s years long!' % (typeOfTrend,trendlength))
        print('-- Number of years is',yearsnew.shape[0],'and number of trends is',slope.shape[0],'--')
        
        if typeOfTrend == 'hiatus':
            SLOPEthresh = meantrend - (1*stdslope)
            obsdiff = abs(np.nanmin(slope) - meantrend)
        elif typeOfTrend == 'accel':
            SLOPEthresh = meantrend + (1*stdslope)
            obsdiff = np.nanmax(slope) - meantrend
        
    elif data.ndim == 2:
        ### Pick start of forced climate change
        yrq = np.where(years[:] >= AGWstart)[0]
        data = data[:,yrq]
        yearsnew = years[yrq]
        print('Years-Trend ---->\n',yearsnew)
        
        ensmean = np.nanmean(data,axis=0)
        yearstrendens = np.empty((len(yearsnew)-trendlength+1,trendlength))
        datatrendens = np.empty((len(yearsnew)-trendlength+1,trendlength))
        for hi in range(len(yearsnew)-(trendlength-1)):
            yearstrendens[hi,:] = np.arange(yearsnew[hi],yearsnew[hi]+trendlength,1)
            datatrendens[hi,:] = ensmean[hi:hi+trendlength]

        ### Calculate trend lines    
        linetrendens = np.empty((len(yearsnew)-trendlength+1,2))
        for hi in range(len(yearsnew)-trendlength+1):
            linetrendens[hi,:] = np.polyfit(yearstrendens[hi],datatrendens[hi],1)
            
        ### Slopes
        slopeens = linetrendens[:,0]
        stdslopeens = np.nanstd(slopeens)
        meantrendens = np.nanmean(slopeens) 
        SLOPEthresh = slopeens
        obsdiff = np.nan
            
    else:
        print(ValueError('WRONG DIMENSIONS OF OBS!'))
        sys.exit()
        
    print('>>>>>>>>>> Ending calc_thresholdOfTrend function!')
    return SLOPEthresh,obsdiff

def calc_HiatusAcc(data,trendlength,years,AGWstart,SLOPEthresh,typeOfTrend,diffBase):
    """
    Function calculates actual trend analysis of hiatus or acceleration in 
    observations and climate model data

    Parameters
    ----------
    data : n-d numpy array
        data from selected data set
    trendlength : integer
        length of trend periods to calculate
    years : 1d array
        Original years of input data
    AGWstart : integer
        Start of data to calculate trends over
    SLOPEthresh : float
        float of the actual trend for hiatus or acceleration
    typeOfTrend : string
        hiatus or accel
    diffBase : float
        percent difference from mean trend trends and obs hiatus/acceleration events
        
    Returns
    -------
    yearstrend : 2-d array
        years calculated for each individual trend line
    linetrend : 2-d array
        slopes and intercepts for each trend line
    indexslopeNegative : n-d array
        index of hiatus or acceleration events
    classes : n-day array
        array of binary numbers for yes event or no event
    
    Usage
    -----
    yearstrend,linetrend,indexslopeNegative,classes = calc_HiatusAcc(data,trendlength,years,AGWstart,SLOPEthresh,typeOfTrend,diffBase)
    """
    print('\n>>>>>>>>>> Using calc_HiatusAcc function!')
    
    ### Import modules
    import numpy as np
    import sys
    
    hiatusSLOPE = SLOPEthresh
                
    if data.ndim == 1:    
        yrq = np.where(years[:] >= AGWstart)[0]
        data = data[yrq]
        yearsnew = years[yrq]
        print('Years-Trend ---->\n',yearsnew)
      
        ### Calculate trend periods
        yearstrend = np.empty((len(yearsnew)-trendlength+1,trendlength))
        datatrend = np.empty((len(yearsnew)-trendlength+1,trendlength))
        for hi in range(len(yearsnew)-(trendlength-1)):
            yearstrend[hi,:] = np.arange(yearsnew[hi],yearsnew[hi]+trendlength,1)
            datatrend[hi,:] = data[hi:hi+trendlength]

        ### Calculate trend lines    
        linetrend = np.empty((len(yearsnew)-trendlength+1,2))
        for hi in range(len(yearsnew)-trendlength+1):         
            linetrend[hi,:] = np.polyfit(yearstrend[hi],datatrend[hi],1)
            
        ### Count number of hiatus or acceleration periods
        slope = linetrend[:,0]     
        if typeOfTrend == 'hiatus':
            indexslopeNegative = np.where((slope[:] <= hiatusSLOPE))[0]
        elif typeOfTrend == 'accel':
            indexslopeNegative = np.where((slope[:] > hiatusSLOPE))[0]
        else:
            print(ValueError('--- WRONG TYPE OF EVENT! ---'))
            sys.exit()
        print('INDEX OF **%s**---->' % typeOfTrend,indexslopeNegative)
        
        ### Calculate classes
        classes = np.zeros((len(yearsnew)))
        classes[indexslopeNegative] = 1
         
    elif data.ndim == 2:
        yrq = np.where(years[:] >= AGWstart)[0]
        data = data[:,yrq]
        yearsnew = years[yrq]
        print('Years-Trend ---->\n',yearsnew)
        
        ens = len(data)
        ### Calculate trend periods
        yearstrend = np.empty((ens,len(yearsnew)-trendlength+1,trendlength))
        datatrend = np.empty((ens,len(yearsnew)-trendlength+1,trendlength))
        for e in range(ens):
            for hi in range(len(yearsnew)-trendlength+1):
                yearstrend[e,hi,:] = np.arange(yearsnew[hi],yearsnew[hi]+trendlength,1)
                datatrend[e,hi,:] = data[e,hi:hi+trendlength]  
                
        ### Calculate trend lines
        linetrend = np.empty((ens,len(yearsnew)-trendlength+1,2))
        for e in range(ens):
            for hi in range(len(yearsnew)-trendlength+1):
                linetrend[e,hi,:] = np.polyfit(yearstrend[e,hi],datatrend[e,hi],1)
        
        ### Count number of hiatus periods
        slope = linetrend[:,:,0]

        if typeOfTrend == 'hiatus':
            indexslopeNegative = []
            for e in range(ens):
                hiatusSLOPEq = hiatusSLOPE-diffBase
                indexslopeNegativeyr = []
                for yr in range(hiatusSLOPEq.shape[0]):
                    if slope[e,yr] <= hiatusSLOPEq[yr]:
                        indexslopeNegativeyr.append(yr)
                indexslopeNegative.append(indexslopeNegativeyr)
        elif typeOfTrend == 'accel':
            indexslopeNegative = []
            for e in range(ens):
                hiatusSLOPEq = hiatusSLOPE+diffBase
                indexslopeNegativeyr = []
                for yr in range(hiatusSLOPEq.shape[0]):
                    if slope[e,yr] > hiatusSLOPEq[yr]:
                        indexslopeNegativeyr.append(yr)
                indexslopeNegative.append(indexslopeNegativeyr)
        else:
            print(ValueError('--- WRONG TYPE OF EVENT! ---'))
            sys.exit()
          
        ### Calculate classes
        classes = np.zeros((slope.shape))
        for e in range(ens):
            classes[e,indexslopeNegative[e]] = 1
     
    print('\n>>>>>>>>>> Ending calc_HiatusAcc function!')                         
    return yearstrend,linetrend,indexslopeNegative,classes