from __future__ import division, print_function, absolute_import 

import argparse
import numpy as np
import amitgroup as ag
import sys

from pnet.bernoullimm import BernoulliMM
import pnet
from sklearn.svm import LinearSVC 

def test(ims, labels, net):
    yhat = net.classify(ims)
    return yhat == labels


if pnet.parallel.main(__name__):
    parser = argparse.ArgumentParser()
    parser.add_argument('model', metavar='<parts net file>', type=argparse.FileType('wb'), help='Filename of model file')


    parser.add_argument('training_seed',metavar= '<param>',type = int)
    parser.add_argument('--log', action='store_true')
    args = parser.parse_args()
    training_seed = args.training_seed
    if args.log:
        from pnet.vzlog import default as vz
    ag.set_verbose(True)
    
    sup_ims = []
    sup_labels = []
    net = None
    layers = [
        #pnet.IntensityThresholdLayer(),
        
        pnet.EdgeLayer(k=5, radius=1, spread='orthogonal', minimum_contrast=0.05),
        pnet.PartsLayer(100, (6, 6), settings=dict(outer_frame=0,
                                                  threshold=40, 
                                                  samples_per_image=40, 
                                                  max_samples=1000000, 
                                                  min_prob=0.005,
                                                  )),
        pnet.ExtensionPartsLayer(num_parts = 100, num_components = 10, part_shape = (12,12), lowerLayerShape = (6,6)),        
        pnet.PoolingLayer(shape=(4,4), strides=(4, 4)),
        #pnet.MixtureClassificationLayer(n_components=1, min_prob=0.0001,block_size=200),
        pnet.SVMClassificationLayer(C=None)
    ]

    net = pnet.PartsNet(layers)

    digits = range(10)
    ims = ag.io.load_mnist('training', selection=slice(10000), return_labels=False)

    net.train(ims)
    
    #sup_ims = []
    #sup_labels = []

    classificationTraining = 10
    rs = np.random.RandomState(training_seed)
    for d in digits:
        ims0 = ag.io.load_mnist('training', [d], return_labels=False)
        indices = np.arange(ims0.shape[0])
        print(indices[:classificationTraining])
        rs.shuffle(indices)
        print(indices[:classificationTraining])
        ims0 = ims0[indices[:classificationTraining]]
        
        sup_ims.append(ims0)
        sup_labels.append(d * np.ones(len(ims0), dtype=np.int64))

    sup_ims = np.concatenate(sup_ims, axis=0)
    sup_labels = np.concatenate(sup_labels, axis=0)

    net.train(sup_ims, sup_labels)


    net.save(args.model)

