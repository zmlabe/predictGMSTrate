"""
Functions calculate hiatus and accleration definitions
 
Notes
-----
    Author : Zachary Labe
    Date   : 17 August 2021
    
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

    Usage
    -----
    SLOPEthresh = calc_thresholdOfTrend(data,trendlength,years,AGWstart,typeOfTrend)
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
        
        print('\n**%s** is %s years long!' % (typeOfTrend,trendlength))
        print('-- Number of years is',yearsnew.shape[0],'and number of trends is',slope.shape[0],'--')
        
        if typeOfTrend == 'hiatus':
            SLOPEthresh = meantrend - (1*stdslope)
        elif typeOfTrend == 'accel':
            SLOPEthresh = meantrend + (1*stdslope) 
    else:
        print(ValueError('WRONG DIMENSIONS OF OBS!'))
        sys.exit()
        
    print('>>>>>>>>>> Ending calc_thresholdOfTrend function!')
    return SLOPEthresh

def calc_HiatusAcc(data,trendlength,years,AGWstart,SLOPEthresh,typeOfTrend):
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
    yearstrend,linetrend,indexslopeNegative,classes = calc_HiatusAcc(data,trendlength,years,AGWstart,SLOPEthresh,typeOfTrend)
    """
    print('\n>>>>>>>>>> Using calc_HiatusAcc function!')
    
    ### Import modules
    import numpy as np
    import sys
    
    hiatusSLOPE = SLOPEthresh
    
    if data.ndim == 2:
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
                indexslopeNegativeq = np.where((slope[e,:] <= hiatusSLOPE))[0]
                if len(indexslopeNegativeq) == 0:
                    indexslopeNegative.append([np.nan])
                else:
                    indexslopeNegative.append(indexslopeNegativeq)
        elif typeOfTrend == 'accel':
            indexslopeNegative = []
            for e in range(ens):
                indexslopeNegativeq = np.where((slope[e,:] > hiatusSLOPE))[0]
                if len(indexslopeNegativeq) == 0:
                    indexslopeNegative.append([np.nan])
                else:
                    indexslopeNegative.append(indexslopeNegativeq)
        else:
            print(ValueError('--- WRONG TYPE OF EVENT! ---'))
            sys.exit()
          
        ### Calculate classes
        classes = np.zeros((data.shape))
        for e in range(ens):
            indexFirstHiatus = []
            for i in range(len(indexslopeNegative[e])):
                if i == 0:
                    saveFirstHiatusYR = indexslopeNegative[e][i]
                    indexFirstHiatus.append(saveFirstHiatusYR)
                elif indexslopeNegative[e][i]-1 != indexslopeNegative[e][i-1]:
                    saveFirstHiatusYR = indexslopeNegative[e][i]
                    indexFirstHiatus.append(saveFirstHiatusYR)
                
            classes[e,indexFirstHiatus] = 1
                
    elif data.ndim == 1:    
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

        print(yearstrend)
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
        indexFirstHiatus = []
        for i in range(len(indexslopeNegative)):
            if i == 0:
                saveFirstHiatusYR = indexslopeNegative[i]
                indexFirstHiatus.append(saveFirstHiatusYR)
            elif indexslopeNegative[i]-1 != indexslopeNegative[i-1]:
                saveFirstHiatusYR = indexslopeNegative[i]
                indexFirstHiatus.append(saveFirstHiatusYR)
        classes = np.zeros((len(yearsnew)))
        classes[indexFirstHiatus] = 1
     
    print('\n>>>>>>>>>> Ending calc_HiatusAcc function!')                         
    return yearstrend,linetrend,indexslopeNegative,classes

def combineEvents(hiatus,accel,typeOfData):
    """
    Function calculates actual trend analysis of hiatus or acceleration in 
    observations and climate model data

    Parameters
    ----------
    hiatus : n-d array
        binary array of hiatus events for all years in data
    accel : n-d array
        binary array of acceleration events for all years in data
    typeOfTrend : string
        hiatus or accel
        
    Returns
    -------
    classEVENTs : n-dy array
        array of classes for both hiatus and acceleration events (0,1,2)

    Usage
    -----
    classEVENTs = combineEvents(hiatus,accel,typeOfData)
    """
    print('\n>>>>>>>>>> Using combineEvents function!')
    
    ### Import modules
    import numpy as np
    import sys
    
    if typeOfData == 'obs':
        zerosAll = np.zeros((hiatus.shape))
        
        whereH = np.where((hiatus == 1.))[0]
        whereA = np.where((accel == 1.))[0]
        zerosAll[whereH] = 1.
        zerosAll[whereA] = 2.
        classEVENTs = zerosAll
        print(typeOfData)
    elif typeOfData == 'model':
        ens = len(hiatus)
        classEVENTs = np.empty((hiatus.shape))
        for e in range(ens):
            zerosAll = np.zeros((hiatus[e].shape))
        
            whereH = np.where((hiatus[e] == 1.))[0]
            whereA = np.where((accel[e] == 1.))[0]
            zerosAll[whereH] = 1.
            zerosAll[whereA] = 2.
            classEVENTs[e,:] = zerosAll
        print(typeOfData)
    else:
        print(ValueError('WRONG TYPE OF DATA SET!'))
        sys.exit()
        
    print('>>>>>>>>>> Ending combineEvents function!')
    return classEVENTs