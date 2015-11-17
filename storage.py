# lshash/storage.py
# Copyright 2012 Kay Zhu (a.k.a He Zhu) and contributors (see CONTRIBUTORS.txt)
#
# This module is part of lshash and is released under
# the MIT License: http://www.opensource.org/licenses/mit-license.php

import json
from sklearn.datasets import fetch_mldata, load_iris, load_digits
import numpy as np
from sklearn import datasets
import cPickle
import gzip
import scipy.io
import sets
import time



try:
    import redis
except ImportError:
    redis = None

__all__ = ['storage']


def storage(storage_config, index):
    """ Given the configuration for storage and the index, return the
    configured storage instance.
    """
    if 'dict' in storage_config:
        return InMemoryStorage(storage_config['dict'])
    elif 'redis' in storage_config:
        storage_config['redis']['db'] = index
        return RedisStorage(storage_config['redis'])
    else:
        raise ValueError("Only in-memory dictionary and Redis are supported.")


class BaseStorage(object):
    def __init__(self, config):
        """ An abstract class used as an adapter for storages. """
        raise NotImplementedError

    def keys(self):
        """ Returns a list of binary hashes that are used as dict keys. """
        raise NotImplementedError

    def set_val(self, key, val):
        """ Set `val` at `key`, note that the `val` must be a string. """
        raise NotImplementedError

    def get_val(self, key):
        """ Return `val` at `key`, note that the `val` must be a string. """
        raise NotImplementedError

    def append_val(self, key, val):
        """ Append `val` to the list stored at `key`.

        If the key is not yet present in storage, create a list with `val` at
        `key`.
        """
        raise NotImplementedError

    def get_list(self, key):
        """ Returns a list stored in storage at `key`.

        This method should return a list of values stored at `key`. `[]` should
        be returned if the list is empty or if `key` is not present in storage.
        """
        raise NotImplementedError


class InMemoryStorage(BaseStorage):
    def __init__(self, config):
        self.name = 'dict'
        self.storage = dict()

    def keys(self):
        return self.storage.keys()

    def set_val(self, key, val):
        self.storage[key] = val

    def get_val(self, key):
        return self.storage[key]

    def append_val(self, key, val):
        # print 'In storage.py'
        # print type(key), type(val)
        # print key
        # print val
        self.storage.setdefault(key, []).append(val)

    def get_list(self, key):
        #------------------------------------------------------------------------------
        # The USPS dataset
        #------------------------------------------------------------------------------
        
        # allPoints, y = datasets.load_digits().data , datasets.load_digits().target
        # digits = fetch_mldata("USPS")
        # X, y = digits.data, digits.target.astype(np.int) - 1
        # allPoints = X
                
        
                
        #------------------------------------------------------------------------------
        # The Digit dataset
        #------------------------------------------------------------------------------
        
        # allPoints, y = datasets.load_digits().data , datasets.load_digits().target
        #------------------------------------------------------------------------------
        
        
        #------------------------------------------------------------------------------
        # The Trajectory dataset
        #------------------------------------------------------------------------------
        # mat = scipy.io.loadmat('CVRR_dataset_trajectory_clustering\i5sim2.mat')
        # # Trajectory dataset loading
        # trajectoriesContainer = []
        # for i in range(1600):
	        # trajectoriesContainer.append([(mat.values()[0][i][0][0][j], mat.values()[0][i][0][1][j]) for j in range(len(mat.values()[0][i][0][0]))])
        # # allPoints = []
        # # for trajectory in trajectoriesContainer:
            # # for point in trajectory:
                # # allPoints.append(point)
#------------------------------------------------------------------------------
        
        
        #------------------------------------------------------------------------------
        # The MINIST dataset
        #------------------------------------------------------------------------------
        # dataset = 'mnist.pkl.gz'
        # f = gzip.open(dataset, 'rb')
        # train_set, valid_set, test_set = cPickle.load(f)
        # allPoints = np.array(list(train_set[0]) + list(valid_set[0]) + list(test_set[0]))
        # allClassLabels = np.array(list(train_set[1]) + list(valid_set[1]) + list(test_set[1]))
        # f.close()
        # print len(allPoints), len(allClassLabels)
        # #------------------------------------------------------------------------------
        # Xlist = []
        # for i in allPoints[:100]:
            # Xlist.append(list(i))	
        # print 'The length of generated keys is : ', len(self.storage.keys())
		# I am trying to print the final list from here because to make sure all points have beed hashed, a better place my be the in the calling python file after the indixing process
        print 'Time before retrieving all indixed trajectories and thier keys : ', time.asctime( time.localtime(time.time()) )
        finalList = []
        fileContainer = open('TrajectoriesClusteringTdrive', 'a')
        for key in self.storage.keys():
            #print type(self.storage[key][0])
            #print type(Xlist[0])
            #print type(np.array(self.storage[key][0]))
            #if len(self.storage[key])>1:
            # # keysWithAssociatedTrajectoryIDs = []
            # # for point in self.storage[key]:
                # # for trajectory in trajectoriesContainer:
                    # # if point in trajectory:
                        # # keysWithAssociatedTrajectoryIDs.append(trajectoriesContainer.index(trajectory))
                        # # break
            # # keysWithAssociatedTrajectoryIDs = sorted(list(set(keysWithAssociatedTrajectoryIDs)))
            finalList.append((len(self.storage[key]), key))#, keysWithAssociatedTrajectoryIDs))#((key, [y[Xlist.index(list(point))] for  point in self.storage[key]]))			
        # # queryResult = []
        count = 0
        for tuple in sorted(finalList):
            # # if 899 in tuple[2]:
                # # queryResult.append(tuple[2])
            count += tuple[0]
            fileContainer.write(str(tuple))
            fileContainer.write('\n')
        fileContainer.write(str(count))
        # # fileContainer.write(str(queryResult))
            #x=[np.array(point) for  point in self.storage[key]]
        #print x[0]
            #print [list(point) for point in self.storage[key][:1]]
        print self.storage[sorted(finalList)[0][1]]
        print 'Time after retrieving all indixed trajectories and thier keys : ', time.asctime( time.localtime(time.time()) )

        return self.storage.get(key, [])


class RedisStorage(BaseStorage):
    def __init__(self, config):
        if not redis:
            raise ImportError("redis-py is required to use Redis as storage.")
        self.name = 'redis'
        self.storage = redis.StrictRedis(**config)

    def keys(self, pattern="*"):
        return self.storage.keys(pattern)

    def set_val(self, key, val):
        self.storage.set(key, val)

    def get_val(self, key):
        return self.storage.get(key)

    def append_val(self, key, val):
        self.storage.rpush(key, json.dumps(val))

    def get_list(self, key):
        return self.storage.lrange(key, 0, -1)
