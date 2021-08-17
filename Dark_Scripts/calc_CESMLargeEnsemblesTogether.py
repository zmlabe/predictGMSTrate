"""
Function reads in both CESM1-LE and CESM2-LE

Notes
-----
    Author : Zachary Labe
    Date   : 28 June 2021

Usage
-----
    [1] read_ALLCESM_le(directory,vari,sliceperiod,slicebase,sliceshape,addclimo,slicenan,takeEnsMean,numOfEns,timeper)
"""

def read_ALLCESM_le(directory,vari,monthlychoice,slicebase,sliceshape,addclimo,slicenan,takeEnsMean,numOfEns,timeper):
    """
    Function reads monthly data from all cesm large ensembles

    Parameters
    ----------
    directory : string
        path for data
    vari : string
        variable for analysis
    sliceperiod : string
        how to average time component of data
    sliceyear : string
        how to slice number of years for data
    sliceshape : string
        shape of output array
    addclimo : binary
        True or false to add climatology
    slicenan : string or float
        Set missing values
    takeEnsMean : binary
        whether to take ensemble mean
    numOfEns : integer
        number of ensemble members to use
    timeper : time period of analysis
        string
    ENSmean : numpy array
        ensemble mean

    Returns
    -------
    lat : 1d numpy array
        latitudes
    lon : 1d numpy array
        longitudes
    var : numpy array
        processed variable

    Usage
    -----
    read_ALLCESM_le(directory,vari,sliceperiod,slicebase,
                    sliceshape,addclimo,slicenan,takeEnsMean,numOfEns,timeper)
    """
    print('\n>>>>>>>>>> STARTING read_ALLCESM_le function!')

    ### Import modules
    import numpy as np
    from netCDF4 import Dataset
    import calc_Utilities as UT
    import read_LENS as LEL
    import read_CESM2LE as CEMLE
    
    ### Parameters 
    directorydataLEL = '/Users/zlabe/Data/LENS/monthly/'
    directorydataCEMLE = '/Users/zlabe/Data/CESM2-LE/monthly/'
    
    ### Read in both large ensembles from cesm2 
    lat1,lon1,cesm1,ENSmean1 = LEL.read_LENS(directorydataLEL,vari,
                                                  monthlychoice,
                                                  slicebase,sliceshape,
                                                  addclimo,slicenan,
                                                  takeEnsMean,timeper) 
    lat1,lon1,cesm2 = CEMLE.read_CESM2LE(directorydataCEMLE,vari,
                                             monthlychoice,sliceshape,
                                             slicenan,numOfEns,timeper)  
    
    ### Combine data 
    models = np.asarray([cesm1,cesm2])

    print('\n\nShape of output FINAL = ', models.shape,[[models.ndim]])
    print('>>>>>>>>>> ENDING read_ALLCESM_le function!')    
    return lat1,lon1,models 

# ### Test functions - do not use!
# import numpy as np
# import matplotlib.pyplot as plt
# import calc_Utilities as UT
# directory = '/Users/zlabe/Data/'
# vari = 'T2M'
# sliceperiod = 'annual'
# slicebase = np.arange(1951,1980+1,1)
# sliceshape = 4
# slicenan = 'nan'
# addclimo = True
# takeEnsMean = False
# timeper = 'historical'
# numOfEns = 40
# lat,lon,var = read_ALLCESM_le(directory,vari,sliceperiod,slicebase,
#                                       sliceshape,addclimo,slicenan,
#                                       takeEnsMean,numOfEns,timeper)
# lon2,lat2 = np.meshgrid(lon,lat)
# ave = UT.calc_weightedAve(var,lat2)
