
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

    error_rates = []
    test_ims, test_labels = ag.io.load_mnist('testing',return_labels=True)
    for i in range(11):
        
        clnet = pnet.PartsNet([net] + [pnet.PoolingLayer(shape=(4,4),strides=(4,4)), pnet.SVMClassificationLayer(C=None)])
        digits = range(10)
        sup_ims = []
        sup_labels = []
        rs = np.random.RandomState(i)
        for d in digits:
            ims0 = ag.io.load_mnist('training',[d],return_labels=False)
            #indices = [k for k in range(len(label)) if label[k] in [d]]
            indices = np.arange(ims0.shape[0])
            print(indices[:10])
            rs.shuffle(indices)
            print(indices[:10])
            sup_ims.append(ims0[indices[:10]])
            sup_labels.append(d * np.ones(10,dtype=np.int64))
        sup_ims = np.concatenate(sup_ims, axis = 0)
        sup_labels = np.concatenate(sup_labels, axis = 0)
       
        clnet.train(sup_ims, sup_labels)

        corrects = 0
        total = 0


        ims_batches = np.array_split(test_ims, 100)
        labels_batches = np.array_split(test_labels, 100)

        def format_error_rate(pr):
            return "{:.2f}%".format(100*(1-pr))

        args = (tup+(clnet,) for tup in itr.izip(ims_batches, labels_batches))
        for i, res in enumerate(pnet.parallel.starmap(test, args)):
            corrects += res.sum()
            total += res.size
            pr = corrects / total

        error_rate = 1.0 - pr
        error_rates.append(error_rate)
    print(error_rates)
    net.save(saveFile)
