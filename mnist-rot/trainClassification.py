
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
    parser.add_argument('data',metavar='<mnist data file>',type=argparse.FileType('rb'), help='Filename of data file')
    parser.add_argument('label', metavar='<mnist data file>', type=argparse.FileType('rb'), help = 'Filename of data label file')

    # there will be #seeds of models that we will train
    parser.add_argument('saveFile', metavar='<name of file to save Model>', type=str, help='Filename of savable model file')

    parser.add_argument('testingData',metavar='<mnist test data file>',type=argparse.FileType('rb'),help='Filename of testing data file')
    parser.add_argument('testingDataLabel', metavar='<mnist test data label file>',type=argparse.FileType('rb'),help='Filename of testing data label')

    # there will be #seeds models that we will train from    
    parser.add_argument('seed', metavar='<training seed>', type=int, help='training seed')

    parser.add_argument('classifier', metavar='<name of the classifier>', type=str, help = 'name of the classifier')
    parser.add_argument('numOfClassModel', metavar='<number of class model>', type=int, help = 'number of class model')
    
    parser.add_argument('rotationOfModel', metavar='<number of rotation in the object model>', type=int, help= 'number of rotations')
    
    parser.add_argument('resultFile', metavar='<name of the accuracy result file>', type=argparse.FileType('wb'), help='Filename of result')
    parser.add_argument('poolingSize', metavar='<size of Pooling>', type=int, help = 'size of pooling')
    parser.add_argument('trainingSize', metavar='<size of training>', type=int, help = 'size of training data')
    parser.add_argument('testingType', metavar='<Type of testing| valicaton| testing>', type = str, help = 'type of testing')

    args = parser.parse_args()
    modelFileName = args.model
    dataFileName = args.data
    dataLabelFileName = args.label
    testingDataFileName = args.testingData
    testingLabelFileName = args.testingDataLabel

    testingData = np.load(testingDataFileName)
    testingLabel = np.load(testingLabelFileName)

    saveFileName = args.saveFile

    data = np.load(dataFileName)
    label = np.load(dataLabelFileName)

    training_seed = args.seed
    classifier = args.classifier
    numOfClassModel = args.numOfClassModel
    resultFile = args.resultFile 
    poolingSize = args.poolingSize
    trainingSize = args.trainingSize
    testingType = args.testingType
    orientations = args.rotationOfModel
    error_rates = []


    validationSet = data[-10000:60000]
    validationLabel = label[-10000:60000]
    


    for i in range(1,training_seed + 1):
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

        net = pnet.PartsNet.load(modelFileName+'%d.npy' % i)
        clnet = pnet.PartsNet([net] + layers)
        
        digits = range(10)
        sup_ims = []
        sup_labels = []
        rs = np.random.RandomState(i)
        for d in digits:
            #ims0 = ag.io.load_mnist('training',[d],return_labels=False)
            indices = [k for k in range(len(label)) if label[k] in [d]]
            ims0 = data[indices]
            indices = np.arange(ims0.shape[0])
            print(indices[:trainingSize])
            rs.shuffle(indices)
            print(indices[:trainingSize])
            sup_ims.append(ims0[indices[:trainingSize]])
            sup_labels.append(d * np.ones(trainingSize,dtype=np.int64))
        sup_ims = np.concatenate(sup_ims, axis = 0)
        sup_labels = np.concatenate(sup_labels, axis = 0)
       
        print('Training supervised...')
        print(clnet._layers) 
        clnet.train(sup_ims, sup_labels)
        print('Done.')


        start0 = time.time()
        print('Training unsupervised...')
        clnet.save(saveFileName + '%d.npy' %i)


        corrects = 0
        total = 0
        if testingType == 'testing':
            test_ims, test_labels = testingData, testingLabel
        elif testingType == 'validation':
            test_ims = validationSet
            print(test_ims.shape)
            test_labels = validationLabel


        ims_batches = np.array_split(test_ims, 100)
        labels_batches = np.array_split(test_labels, 100)

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
    np.save(resultFile, error_rates)
    print(error_rates)
