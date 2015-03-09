
from __future__ import division, print_function, absolute_import

#from pnet.vzlog import default as vz
import numpy as np
import amitgroup as ag
import itertools as itr
import sys
import os


import pnet
import time

def test(ims, labels, net):
    yhat = net.classify(ims)
    return yhat == labels

if pnet.parallel.main(__name__):

    ag.set_verbose(True)
    import argparse
    parser = argparse.ArgumentParser()
    #there will be #seeds models that we will train from, thus we will have #seeds saved models and saved result

    parser.add_argument('model', metavar='<model file>', type=str, help='Filename of model file')

    parser.add_argument('classifier', metavar='<name of the classifier>', type=str, help = 'name of the classifier')
    parser.add_argument('numOfClassModel', metavar='<number of class model>', type=int, help = 'number of class model')
    
    parser.add_argument('rotationOfModel', metavar='<number of rotation in the object model>', type=int, help= 'number of rotations')
    parser.add_argument('poolingSize', metavar='<size of Pooling>', type=int, help = 'size of pooling')

    args = parser.parse_args()
    modelFileName = args.model
    classifier = args.classifier
    numOfClassModel = args.numOfClassModel
    poolingSize = args.poolingSize
    orientations = args.rotationOfModel
    error_rates = []

    if classifier == 'mixture':
        print('mixture')
        cl = pnet.MixtureClassificationLayer(n_components=numOfClassModel, min_prob=1e-5, block_size=200)
    elif classifier == 'svm':
        print('svm')
        cl = pnet.SVMClassificationLayer(C=None)
    elif classifier == 'rot-mixture':
        cl = pnet.RotationMixtureClassificationLayer(n_components=numOfClassModel, n_orientations=orientations, min_prob=0.0001, pooling_settings=dict(shape=(poolingSize,poolingSize),strides=(poolingSize,poolingSize),rotation_spreading_radius=0))

    if (poolingSize != 0) and (classifier != 'rot-mixture'):
        layers = [pnet.PoolingLayer(shape=(poolingSize,poolingSize),strides=(poolingSize, poolingSize)), cl]
    else:
        layers = [cl]

    net = pnet.PartsNet.load(modelFileName)
    clnet = pnet.PartsNet([net] + layers)
    
    digits = range(10)
    sup_ims = []
    sup_labels = []
    rs = np.random.RandomState(i)
    for d in digits:
        ims0,tmpLabel = ag.io.load_cifar10('training',[d])
        indices = np.arange(ims0.shape[0])
        sup_ims.append(ims0)
        sup_labels.append(d * np.ones(len(ims0),dtype=np.int64))
    sup_ims = np.concatenate(sup_ims, axis = 0)
    sup_labels = np.concatenate(sup_labels, axis = 0)
   
    print('Training supervised...')
    print(clnet._layers) 
    clnet.train(sup_ims, sup_labels)
    print('Done.')

    start0 = time.time()

    corrects = 0
    total = 0
    test_ims, test_labels = ag.io.load_cifar10('testing')

    ims_batches = np.array_split(test_ims, 200)
    labels_batches = np.array_split(test_labels, 200)

    def format_error_rate(pr):
        return "{:.2f}%".format(100*(1-pr))

    args = (tup+(clnet,) for tup in itr.izip(ims_batches, labels_batches))
    for i, res in enumerate(pnet.parallel.starmap(test, args)):
        if i!=0 and i %20 ==0:
            print("{0:05}/{1:05} Error rate: {2}".format(total, len(ims0),format_error_rate(pr)))
        corrects += res.sum()
        total += res.size
        pr = corrects / total

    print("Final error rate:", format_error_rate(pr))
    error_rate = 1.0 - pr
    error_rates.append(error_rate)
    print(error_rates)
