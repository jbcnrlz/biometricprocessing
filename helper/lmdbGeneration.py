import lmdb, numpy as np, h5py

def load_data_into_lmdb(lmdb_name, features, labels=None):
    env = lmdb.open(lmdb_name, map_size=features.nbytes*2)
    
    features = features[:,:,None,None]
    for i in range(features.shape[0]):
        datum = caffe.proto.caffe_pb2.Datum()
        
        datum.channels = features.shape[1]
        datum.height = 1
        datum.width = 1
        
        if features[i].dtype == np.int:
            datum.data = features[i].tostring()
        elif features[i].dtype == np.float: 
            datum.float_data.extend(features[i].flat)
        else:
            raise Exception("features.dtype unknown.")
        
        if labels is not None:
            datum.label = int(labels[i])
        
        str_id = '{:08}'.format(i)
        with env.begin(write=True) as txn:
            txn.put(str_id.encode('ascii'), datum.SerializeToString())

def h5Format(pathh5,features,labels,imageMean = None):
    if (len(features.shape) >= 4):
        X = np.zeros((features.shape[0],features.shape[3],features.shape[1],features.shape[2]))
    elif (len(features.shape) == 3):
        X = np.zeros((features.shape[0],1,features.shape[1],features.shape[2]))
    else:
        X = np.zeros((features.shape[0],1,features.shape[1],features.shape[2]))
    Y = np.zeros((len(labels),1,1,1))
    for t in range(len(features)):
        if (len(features[t].shape) >= 3):
            X[t] = features[t].transpose((2,0,1))
        else:
            X[t] = features[t]
            
        Y[t] = labels[t].reshape((1,1,1))

    if (imageMean != None):
        X = X - imageMean

    with h5py.File(pathh5,'w') as H:
        H.create_dataset( 'data', data=X )
        H.create_dataset( 'label', data=Y )
    with open(pathh5[:-3]+'_list.txt','w') as L:
        rp = pathh5.split('/')
        L.write( rp[-1] )