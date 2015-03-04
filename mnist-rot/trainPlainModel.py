
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

    parser.add_argument('numParts', metavar='<numPart param>', type=int, help='number of parts')
    parser.add_argument('patchSize', metavar='<patch size>', type=int, help='size of patches')
    parser.add_argument('data',metavar='<mnist data file>',type=argparse.FileType('rb'), help='Filename of data file')
    parser.add_argument('saveFile', metavar='<part Model file>', type=argparse.FileType('wb'),help='Filename of savable model file')

    parser.add_argument('seed', metavar='<training seed>', type=int, help='training seed')

    args = parser.parse_args()
    numParts = args.numParts
    dataFileName = args.data
    saveFile = args.saveFile
    patchSize = args.patchSize
    data = np.load(dataFileName)
    training_seed = args.seed
    
    print("Inside")
    layers = [
        pnet.EdgeLayer(k=5, radius=1, spread='orthogonal', minimum_contrast=0.05),#

                pnet.PartsLayer(numParts, (patchSize, patchSize), settings=dict(outer_frame=0,
                                                  #em_seed=training_seed,
                                                  threshold=40,
                                                  samples_per_image=40,
                                                  max_samples=1000000,
                                                  min_prob=0.005,
                                                  )),
    ]

    net = pnet.PartsNet(layers)

    digits = range(10)
    print('Extracting subsets...')

    ims10k = data[:10000]
    print('Done.')


    start0 = time.time()
    print('Training unsupervised...')
    net.train(ims10k)
    print('Done.')
    end0 = time.time()

    net.save(saveFile)
