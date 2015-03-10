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
    parser.add_argument('lowerPartsModel',metavar='<model file>', type = argparse.FileType('rb'), help='Filename of model file')
    parser.add_argument('numParts',metavar='<numPart param>',type = int, help='number of parts')
    parser.add_argument('numExtensionParts', metavar='<number of extension parts>', type = int, help = 'number of extension parts for each low level parts')
    parser.add_argument('extensionPatchSize',metavar='<patch Size>',type = int, help = 'size of patches')
    parser.add_argument('model', metavar='<parts net file>', type=argparse.FileType('wb'), help='Filename of model file')
    parser.add_argument('--log', action='store_true')
    args = parser.parse_args()
    

    numParts = args.numParts
    numExtensionParts = args.numExtensionParts
    extensionPatchSize = args.extensionPatchSize

    if args.log:
        from pnet.vzlog import default as vz
    ag.set_verbose(True)
    
    sup_ims = []
    sup_labels = []
    net = pnet.PartsNet.load(args.lowerPartsModel)
    layers = [
        #pnet.IntensityThresholdLayer(),
        pnet.ExtensionPartsLayer(num_parts = numParts, num_components = numExtensionParts, part_shape = (extensionPatchSize, extensionPatchSize), lowerLayerShape = (6,6)) 
    ]
    clnet = pnet.PartsNet([net] + layers)


    ims, label = ag.io.load_cifar10('training')

    clnet.train(ims)
    clnet.save(args.model)

'''
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
'''
