"""
Function to slice trainging/testing/validation data
 
Notes
-----
    Author  : Zachary Labe
    Date    : 13 September 2021
    Version : 2  (added validation data slice)
    
Usage
-----
    [1] segment_data(data,classesl,fac,random_segment_seed)
"""
def segment_data(data,classesl,fac,random_segment_seed):
    """
    Function to segment data based on ensemble members

    Parameters
    ----------
    data : n-d numpy array
        data from selected data set
    classesl : n-d array = same as data
        processed hiatus binary classification
    fac : float
        fraction to use for training/testing ensembles
    random_segment_seed : integer
        random seed to select ensembles
        
    Returns
    -------
    Xtrain : n-d array
        training ensembles data
    Ytrain : n-d array
        classes for training ensembles data
    Xtest : n-d array
        testing ensembles data
    Ytest : n-d array
        classes for testing ensembles data
    Xtval : n-d array
        validation ensembles data
    Yval : n-d array
        classes for validation ensembles data
    Xtrain_shape : shape
        shape of training array
    Xtest_shape : shape
        shape of testing array
    Xval_shape : shape
        shape of validation array
    testIndices : 1-d array
        list of testing ensembles
    trainIndices : 1-d array
        list of training ensembles
    valIndices : 1-d array
        list of validation ensembles
    class_weight : dictionary
        weights for each class
    random_segment_seed : integer
        random seed to select ensembles

    Usage
    -----
    segment_data(data,classesl,fac,random_segment_seed)
    """
    print('\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Using segment_data function!\n')
    
    ### Import modules
    import numpy as np
    from tensorflow.keras.utils import to_categorical
    import sys
    
    ### Create class weights
    def class_weight_creator(Y):
        class_dict = {}
        weights = np.max(np.sum(Y, axis=0)) / np.sum(Y, axis=0)
        for i in range( Y.shape[-1] ):
            class_dict[i] = weights[i]               
        return class_dict
          
    ### Random seeds for segmenting ensembles
    if random_segment_seed == None:
        random_segment_seed = int(int(np.random.randint(1, 100000)))
    np.random.seed(random_segment_seed)
    
    ############################################################################### 
    ############################################################################### 
    ###############################################################################             
    ###################################################################
    ### Large Ensemble experiment
        
    ### Flip GCM and ensemble member axes
    datanew = np.swapaxes(data,0,1)
    classeslnew = np.swapaxes(classesl,0,1)

    if fac < 1 :
        nrows = datanew.shape[0]
        segment_train = int(np.round(nrows * fac))
        segment_test = int(np.round(nrows * 0.2))
        segment_val = nrows - segment_train - segment_test
        print('--------------------------------------------------------------------')
        print('Training on',segment_train,'ensembles, Testing on',segment_test,'ensembles, Validation on',segment_val,'ensembles')
        print('--------------------------------------------------------------------')

        ### Picking out random ensembles
        i = 0
        trainIndices = list()
        while i < segment_train:
            line = np.random.randint(0, nrows)
            if line not in trainIndices:
                trainIndices.append(line)
                i += 1
            else:
                pass
    
        i = 0
        testIndices = list()
        while i < segment_test:
            line = np.random.randint(0, nrows)
            if line not in trainIndices:
                if line not in testIndices:
                    testIndices.append(line)
                    i += 1
            else:
                pass
            
        i = 0
        valIndices = list()
        while i < segment_val:
            line = np.random.randint(0, nrows)
            if line not in trainIndices:
                if line not in testIndices:
                    if line not in valIndices:
                        valIndices.append(line)
                        i += 1
            else:
                pass

###############################################################################  
###############################################################################  
###############################################################################  
        ### Training segment----------
        data_train = np.empty((len(trainIndices),datanew.shape[1],
                                datanew.shape[2],datanew.shape[3],
                                datanew.shape[4]))
        Ytrain = np.empty((len(trainIndices),classeslnew.shape[1],
                            classeslnew.shape[2]))
        for index,ensemble in enumerate(trainIndices):
            data_train[index,:,:,:,:] = datanew[ensemble,:,:,:,:]
            Ytrain[index,:,:] = classeslnew[ensemble,:,:]
            
        ### Random ensembles are picked
        print('\n----------------------------------------')
        print('Training on ensembles: ',trainIndices)
        print('Testing on ensembles: ',testIndices)
        print('Validation on ensembles: ',valIndices)
        print('----------------------------------------')
        print('\n----------------------------------------')
        print('org data - shape', datanew.shape)
        print('training data - shape', data_train.shape)
    
        ### Reshape into X and Y
        Xtrain = data_train.reshape((data_train.shape[0]*data_train.shape[1]*data_train.shape[2]),(data_train.shape[3]*data_train.shape[4]))
        Ytrain = Ytrain.reshape((Ytrain.shape[0]*Ytrain.shape[1]*Ytrain.shape[2]))
        Xtrain_shape = (data_train.shape[0],data_train.shape[1])

###############################################################################  
###############################################################################          
###############################################################################        
        ### Testing segment----------
        data_test = np.empty((len(testIndices),datanew.shape[1],
                                datanew.shape[2],datanew.shape[3],
                                datanew.shape[4]))
        Ytest = np.empty((len(testIndices),classeslnew.shape[1],
                            classeslnew.shape[2]))
        for index,ensemble in enumerate(testIndices):
            data_test[index,:,:,:,:] = datanew[ensemble,:,:,:,:]
            Ytest[index,:,:] = classeslnew[ensemble,:,:]
        
        ### Random ensembles are picked
        print('----------------------------------------\n')
        print('----------------------------------------')
        print('Training on ensembles: count %s' % len(trainIndices))
        print('Testing on ensembles: count %s' % len(testIndices))
        print('Validation on ensembles: count %s' % len(valIndices))
        print('----------------------------------------\n')
        
        print('----------------------------------------')
        print('org data - shape', datanew.shape)
        print('testing data - shape', data_test.shape)
        print('----------------------------------------')

        ### Reshape into X and Y
        Xtest= data_test.reshape((data_test.shape[0]*data_test.shape[1]*data_test.shape[2]),(data_test.shape[3]*data_test.shape[4]))
        Ytest = Ytest.reshape((Ytest.shape[0]*Ytest.shape[1]*Ytest.shape[2]))
        Xtest_shape = (data_test.shape[0],data_test.shape[1])
        
###############################################################################  
###############################################################################  
###############################################################################  
        ### Validation segment----------
        data_val = np.empty((len(valIndices),datanew.shape[1],
                                datanew.shape[2],datanew.shape[3],
                                datanew.shape[4]))
        Yval = np.empty((len(valIndices),classeslnew.shape[1],
                            classeslnew.shape[2]))
        for index,ensemble in enumerate(valIndices):
            data_val[index,:,:,:,:] = datanew[ensemble,:,:,:,:]
            Yval[index,:,:] = classeslnew[ensemble,:,:]
        
        ### Random ensembles are picked
        print('\n----------------------------------------')
        print('Training on ensembles: count %s' % len(trainIndices))
        print('Testing on ensembles: count %s' % len(testIndices))
        print('Validation on ensembles: count %s' % len(valIndices))
        print('----------------------------------------\n')
        print('----------------------------------------')
        print('org data - shape', datanew.shape)
        print('Validation data - shape', data_val.shape)
        print('----------------------------------------')

        ### Reshape into X and Y
        Xval= data_val.reshape((data_val.shape[0]*data_val.shape[1]*data_val.shape[2]),(data_val.shape[3]*data_val.shape[4]))
        Yval = Yval.reshape((Yval.shape[0]*Yval.shape[1]*Yval.shape[2]))
        Xval_shape = (data_val.shape[0],data_val.shape[1])
      
        ### 'unlock' the random seed
        np.random.seed(None)
        
        ### One-hot vectors
        Ytrain = to_categorical(Ytrain)
        Ytest = to_categorical(Ytest)  
        Yval = to_categorical(Yval)  
        
        ### Class weights
        class_weight = class_weight_creator(Ytrain)
    
    print('\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Ending segment_data function!')  
    return Xtrain,Ytrain,Xtest,Ytest,Xval,Yval,Xtrain_shape,Xtest_shape,Xval_shape,testIndices,trainIndices,valIndices,class_weight,random_segment_seed