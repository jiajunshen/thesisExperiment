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
    parser.add_argument('model', metavar='<model file>', type=argparse.FileType('rb'), help='Filename of model file')
    parser.add_argument('numParts', metavar='<numPart param>', type=int, help='number of parts')
    parser.add_argument('numExtensionParts', metavar='<number of extension parts>', type=int, help='number of extension parts for each low level parts')
    parser.add_argument('extensionPatchSize', metavar='<patch size>', type=int, help='size of patches')
    parser.add_argument('data',metavar='<mnist data file>',type=argparse.FileType('rb'), help='Filename of data file')
    parser.add_argument('saveFile', metavar='<part Model file>', type=argparse.FileType('wb'),help='Filename of savable model file')
    parser.add_argument('seed', metavar='<training seed>', type=int, help='training seed')

    args = parser.parse_args()
    model = args.model
    numParts = args.numParts
    numExtensionParts = args.numExtensionParts
    dataFileName = args.data
    saveFile = args.saveFile
    extensionPatchSize = args.extensionPatchSize
    data = np.load(dataFileName)
    training_seed = args.seed
    net = pnet.PartsNet.load(args.model)

    print("Inside")
    extensionlayers = [
        pnet.ExtensionPartsLayer(num_parts = numParts, num_components = numExtensionParts, part_shape = (extensionPatchSize,extensionPatchSize), lowerLayerShape = (6,6))
    ]
    clnet = pnet.PartsNet([net] + extensionlayers)

    digits = range(10)
    print('Extracting subsets...')

    ims10k = data[:10000]
    print('Done.')


    start0 = time.time()
    print('Training unsupervised...')
    clnet.train(ims10k)
    print('Done.')
    end0 = time.time()

    clnet.save(saveFile)
