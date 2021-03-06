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
    pnet.OrientedPartsLayer(16,4,(6,6),settings=dict(outer_frame = 1,
                                em_seed=training_seed,
                                n_init = 2,
                                threshold = 35,
                                samples_per_image=40,
                                max_samples=50000,
                                train_limit=20000,
                                rotation_spreading_radius = 0,
                                min_prob= 0.005,
                                edge_type = 'colorYali',
                                bedges = dict(k = 5,
                                               minimum_contrast = 0.05,
                                               spread = 'orthogonal',
                                                radius = 1,
                                                contrast_insensitive=False,
                                               ),
    )), 
        
        #pnet.PoolingLayer(shape=(4,4), strides=(4, 4)),
        #pnet.MixtureClassificationLayer(n_components=1, min_prob=0.0001,block_size=200),
        pnet.RotExtensionPartsLayer(num_parts = 16, num_components = 5, rotation = 4, part_shape = (12,12),lowerLayerShape=(6,6)),
        #pnet.ExtensionPartsLayer(num_parts = 128, num_components = 5, part_shape = (12,12),lowerLayerShape = (6,6)),
        #pnet.ExtensionPoolingLayer(n_parts = 1000, grouping_type = 'rbm', pooling_type = 'distance', pooling_distance = 5, weights_file = None, save_weights_file = None, settings = {}) 
        pnet.PoolingLayer(shape=(8,8), strides=(8, 8)),
        pnet.SVMClassificationLayer(C=None)
    ]

    net = pnet.PartsNet(layers)

    ims, label = ag.io.load_cifar10('training', selection = slice(0,10000))

    net.train(ims)
    
    classes = range(10)
    classificationTraining = 5000
    rs = np.random.RandomState(training_seed)
    for d in classes:
        ims0,tmpLabel = ag.io.load_cifar10('training', classes = [d])
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

