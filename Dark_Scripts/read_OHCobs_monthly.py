"""
Function reads in data from NOAA or IAP ocean heat content data sets
 
Notes
-----
    Author : Zachary Labe
    Date   : 26 August 2021
    
Usage
-----
    [1] read_OHCobs(variq,directory,ohcDATA,sliceperiod,sliceyear,sliceshape,slicenan)
"""

def read_OHCobs(variq,directory,ohcDATA,sliceperiod,sliceyear,sliceshape,slicenan):
    """
    Function reads monthly data from ERA5
    
    Parameters
    ----------
    variq : string
        variable to retrieve
    directory : string
        path for data
    ohcDATA : string
        which data set to use for OHC
    sliceperiod : string
        how to average time component of data
    sliceyear : string
        how to slice number of years for data
    sliceshape : string
        shape of output array
    slicenan : string or float
        Set missing values
        
    Returns
    -------
    lat : 1d numpy array
        latitudes
    lon : 1d numpy array
        longitudes
    var : 3d numpy array or 4d numpy array 
        [annual,lat,lon] or [year,month,lat,lon]
        
    Usage
    -----
    lat,lon,var = read_OHCobs(variq,directory,ohcDATA,sliceperiod,sliceyear,
                              sliceshape,slicenan)
    """
    print('>>>>>>>>>> STARTING read_OHCobs function! -- %s' % variq)
    
    ### Import modules
    import numpy as np
    from netCDF4 import Dataset
    import warnings
    import calc_Utilities as UT
    import sys
    warnings.simplefilter(action='ignore', category=FutureWarning)
    warnings.simplefilter(action='ignore', category=RuntimeWarning)
    
    ###########################################################################
    ### Read in data
    if ohcDATA == 'IAP':
        time = np.arange(1940,2020+1,1)
        mon = 12
        filename = 'OHC-IAP/monthly/%s_1940-2020.nc' % variq
        data = Dataset(directory + filename,'r')
        lat1 = data.variables['latitude'][:]
        lon1 = data.variables['longitude'][:]
        var = data.variables['%s' % variq][:,:,:]
        data.close()
    elif ohcDATA == 'NOAANCEI':
        if variq == 'OHC700':
            time = np.arange(1955,2020+1,1)
            mon = 12
            filename = 'OHC-NOAANCEI/yearly/%s_1955-2020.nc' % variq
            data = Dataset(directory + filename,'r')
            lat1 = data.variables['latitude'][:]
            lon1 = data.variables['longitude'][:]
            var = data.variables['%s' % variq][:,:,:,:].squeeze()
            data.close()
        elif variq == 'OHC2000':
            time = np.arange(2005,2020+1,1)
            mon = 12
            filename = 'OHC-NOAANCEI/yearly/%s_2005-2020.nc' % variq
            data = Dataset(directory + filename,'r')
            lat1 = data.variables['latitude'][:]
            lon1 = data.variables['longitude'][:]
            var = data.variables['%s' % variq][:,:,:,:].squeeze()
            data.close()
    else:
        print(ValueError('WRONG OHC DATA SET!'))
        sys.exit()
    
    print('Years of output =',time.min(),'to',time.max())
    ###########################################################################
    ### Reshape data into [year,month,lat,lon]
    if ohcDATA != 'NOAANCEI':
        varmon  = np.reshape(var,(var.shape[0]//mon,mon,
                                   lat1.shape[0],lon1.shape[0]))
    else:
        varmon = var
    
    ###########################################################################
    ### Slice over months (currently = [yr,mn,lat,lon])
    ### Shape of output array
    if ohcDATA == 'IAP':
        if sliceperiod == 'annual':
            vartime = np.nanmean(varmon,axis=1)
            if sliceshape == 1:
                varshape = vartime.ravel()
            elif sliceshape == 3:
                varshape = vartime
            print('Shape of output = ', varshape.shape,[[varshape.ndim]])
            print('Completed: ANNUAL MEAN!')
        elif sliceperiod == 'DJF':
            varshape = UT.calcDecJanFeb(varmon,lat1,lon1,'surface',1)
            print('Shape of output = ', varshape.shape,[[varshape.ndim]])
            print('Completed: DJF MEAN!')
        elif sliceperiod == 'JJA':
            vartime = np.nanmean(varmon[:,5:8,:,:],axis=1)
            if sliceshape == 1:
                varshape = vartime.ravel()
            elif sliceshape == 3:
                varshape = vartime
            print('Shape of output = ', varshape.shape,[[varshape.ndim]])
            print('Completed: JJA MEAN!')
        elif sliceperiod == 'JFM':
            vartime = np.nanmean(varmon[:,0:3,:,:],axis=1)
            if sliceshape == 1:
                varshape = vartime.ravel()
            elif sliceshape == 3:
                varshape = vartime
            print('Shape of output = ', varshape.shape,[[varshape.ndim]])
            print('Completed: JFM MEAN!')
        elif sliceperiod == 'AMJ':
            vartime = np.nanmean(varmon[:,3:6,:,:],axis=1)
            if sliceshape == 1:
                varshape = vartime.ravel()
            elif sliceshape == 3:
                varshape = vartime
            print('Shape of output = ', varshape.shape,[[varshape.ndim]])
            print('Completed: AMJ MEAN!')
        elif sliceperiod == 'JAS':
            vartime = np.nanmean(varmon[:,6:9,:,:],axis=1)
            if sliceshape == 1:
                varshape = vartime.ravel()
            elif sliceshape == 3:
                varshape = vartime
            print('Shape of output = ', varshape.shape,[[varshape.ndim]])
            print('Completed: JAS MEAN!')
        elif sliceperiod == 'OND':
            vartime = np.nanmean(varmon[:,9:,:,:],axis=1)
            if sliceshape == 1:
                varshape = vartime.ravel()
            elif sliceshape == 3:
                varshape = vartime
            print('Shape of output = ', varshape.shape,[[varshape.ndim]])
            print('Completed: OND MEAN!')
        elif sliceperiod == 'none':
            vartime = varmon
            if sliceshape == 1:
                varshape = vartime.ravel()
            elif sliceshape == 4:
                varshape = varmon
    else: 
        vartime = varmon
        if sliceshape == 1:
            varshape = varshape.ravel()
        elif sliceshape == 3:
            varshape = varmon
    print('Shape of output =', varshape.shape, [[varshape.ndim]])
    print('Completed: ALL MONTHS!')
        
    ###########################################################################
    ### Change missing values
    if slicenan == 'nan':
        varshape[np.where(np.isnan(varshape))] = np.nan
        print('Completed: missing values are =',slicenan)
    else:
        varshape[np.where(np.isnan(varshape))] = slicenan
        
    ###########################################################################
    ### Select years of analysis (1979-2020)
    if ohcDATA != 'NOAANCEI':
        yearhistq = np.where((time >= 1979) & (time <= 2020))[0]
        print('1979-2020')
        print(time[yearhistq])
        varshapetime = varshape[yearhistq,:,:]
    else:
        if variq == 'OHC700':
            yearhistq = np.where((time >= 1979) & (time <= 2020))[0]
            print('1979-2020')
            print(time[yearhistq])
            varshapetime = varshape[yearhistq,:,:]
        elif variq == 'OHC2000':
            varshapetime = varshape
            print('2005-2020')
        
    print('>>>>>>>>>> ENDING read_OHCobs function! -- %s' % variq)
    return lat1,lon1,varshapetime

# ### Test functions - do not use!
# import numpy as np
# import matplotlib.pyplot as plt
# import calc_Utilities as UT
# variq = 'OHC300'
# directory = '/Users/zlabe/Data/'
# sliceperiod = 'annual'
# sliceyear = np.arange(1940,2020+1,1)
# sliceshape = 3
# slicenan = 'nan'
# ohcDATA = 'IAP'
# lat,lon,var = read_OHCobs(variq,directory,ohcDATA,sliceperiod,
#                                 sliceyear,sliceshape,slicenan)
# lon2,lat2 = np.meshgrid(lon,lat)
# ave = UT.calc_weightedAve(var,lat2)