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
    parser.add_argument('poolingShape', metavar = '<pooling patch size>', type=int, help='size of pooling')
    parser.add_argument('data',metavar='<mnist data file>',type=argparse.FileType('rb'), help='Filename of data file')
    parser.add_argument('saveFile', metavar='<part Model file>', type=argparse.FileType('wb'),help='Filename of savable model file')
    parser.add_argument('saveWeightsFile', metavar='<pooling weights file>', type=argparse.FileType('wb'), help='Filename of weights file')
    parser.add_argument('poolingDistance',metavar='<distance of pooling>', type=int,help='Distance of Pooling')
    parser.add_argument('seed', metavar='<training seed>', type=int, help='training seed')
    parser.add_argument('new_seed',metavar = '<random seed>', type=int, help = 'random seed')
    parser.add_argument('epochs',metavar = '<number of epochs>', type=int, help = 'random seed')
    parser.add_argument('n_hidden',metavar = '<number of hidden nodes>', type=int, help = 'random seed')
    parser.add_argument('rate',metavar = '<rate>', type=int, help = 'random seed')
    args = parser.parse_args()
    model = args.model
    numParts = args.numParts
    numExtensionParts = args.numExtensionParts
    dataFileName = args.data
    saveFile = args.saveFile
    weightsSaveFile = args.saveWeightsFile
    extensionPatchSize = args.extensionPatchSize
    data = np.load(dataFileName)
    training_seed = args.seed
    random_seed = args.new_seed
    net = pnet.PartsNet.load(args.model)
    pooling_distance = args.poolingDistance
    pooling_shape = args.poolingShape
    epochs = args.epochs
    n_hidden = args.n_hidden
    rate = args.rate/100.0

    print(model, numParts, numExtensionParts, dataFileName, saveFile, weightsSaveFile, extensionPatchSize, training_seed, pooling_distance, pooling_shape)
    print("Inside")
    extensionlayers = [
        #pnet.ExtensionPartsLayer(num_parts = numParts, num_components = numExtensionParts, part_shape = (extensionPatchSize,extensionPatchSize), lowerLayerShape = (6,6))
        pnet.PoolingLayer(shape=(pooling_shape,pooling_shape),strides=(pooling_shape,pooling_shape)),
        pnet.ExtensionPoolingLayer(n_parts = numParts * numExtensionParts, grouping_type='rbm', pooling_type='distance',pooling_distance=pooling_distance, save_weights_file = weightsSaveFile, weights_file = None, settings=dict(n_hidden = n_hidden, epochs = epochs, rate = rate, rbm_seed = random_seed))
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
