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
        pnet.OrientedGaussianPartsLayer(n_parts = 32, n_orientations = 1, part_shape = (5,5), settings= dict(n_iter = 10,
                            seed=0,
                            n_init=1,
                            standardize=True,
                            samples_per_image=40,
                            #max_samples=300000,
                            max_samples=100000,
                            uniform_weights=True,
                            max_covariance_samples=None,
                            covariance_type='tied',
                            logratio_thresh=-np.inf,
                            std_thresh_frame=0,
                            channel_mode='together',
                            coding='hard',
                            normalize_globally=False,
                            min_covariance=1e-3,
                            std_thresh=0.01,
                            standardization_epsilon=0.001,
                            code_bkg=False,
                            whitening_epsilon=None,
                            min_count=0,
                            )),
        pnet.PoolingLayer(shape=(1,1), strides=(1, 1)),
        #pnet.ColorEdgeLayer(k=5, radius=1, spread='orthogonal', minimum_contrast=0.05),
        pnet.PartsLayer(100, (6, 6), settings=dict(outer_frame=0,
                                                  threshold=20, 
                                                  samples_per_image=20, 
                                                  max_samples=1000000, 
                                                  min_prob=0.005,
                                                  )),
        
        #pnet.MixtureClassificationLayer(n_components=1, min_prob=0.0001,block_size=200),
        pnet.ExtensionPartsLayer(num_parts = 100, num_components = 10, part_shape = (12,12),lowerLayerShape = (6,6)),
        #pnet.ExtensionPoolingLayer(n_parts = 1000, grouping_type = 'rbm', pooling_type = 'distance', pooling_distance = 5, weights_file = None, save_weights_file = None, settings = {}) 
        pnet.PoolingLayer(shape=(4,4), strides=(4, 4)),
        pnet.SVMClassificationLayer(C=None)
    ]

    net = pnet.PartsNet(layers)

    ims, label = ag.io.load_cifar10('training', selection = slice(0,10000))

    net.train(ims)
    print("===============")
    print(net.layers[0]._means.shape)
    classes = range(10)
    classificationTraining = 10
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

