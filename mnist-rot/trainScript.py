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
    print("1")
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('model',metavar='<model file>', type=argparse.FileType('rb'), help='Filename of the model file')
    parser.add_argument('data',metavar='<mnist data file>',type=argparse.FileType('rb'), help='Filename of data file')
    parser.add_argument('label',metavar='<mnist label file>', type = argparse.FileType('rb'), help = 'Filename of label file')
    parser.add_argument('classifier',metavar='<classifier>', type=str, choices=('svm-mixture', 'rot-mixture'), help='num Of Class Model')
    parser.add_argument('numOfClassModel',metavar='<numOfClassModel>', type=str, help='num Of Class Model')

    args = parser.parse_args()

    numOfClassModel = args.numOfClassModel

    if numOfClassModel == 'many':
        num_class_models = [1, 2, 5, 10, 20]
    else:
        num_class_models = [int(numOfClassModel)]

    if args.classifier == 'svm-mixture':
        classifier_names = ['mixture', 'svm']

    data = np.load(args.data)
    label = np.load(args.label)
    net = pnet.PartsNet.load(args.model)

    unsup_training_times = []
    sup_training_times = []
    testing_times = []
    error_rates = []
    all_num_parts = []

    ims10k = data[:10000]
    label10k = np.array(label[:10000])
    ims2k = data[10000:12000]
    label2k = np.array(label[10000:12000])


    digits = range(10)

    sup_ims = ims10k
    sup_labels = label10k

    for classifier in classifier_names:
        num_class_models0 = num_class_models
        if classifier == 'svm':
            num_class_models0 = [1]

        for n_classes in num_class_models0:
            print('Classifier:', classifier, 'Components:', n_classes)
            if classifier == 'mixture':
                layers = [
                    pnet.PoolingLayer(shape=(4, 4), strides=(4, 4)),
                    pnet.MixtureClassificationLayer(n_components=n_classes, min_prob=1e-5)
                ]
            elif classifier == 'svm':
                layers = [
                    pnet.PoolingLayer(shape=(4, 4), strides=(4, 4)),
                    pnet.SVMClassificationLayer(C=None),
                ]

            clnet = pnet.PartsNet([net] + layers)

            start1 = time.time()
            print('Training supervised...')
            print(sup_ims.shape)
            clnet.train(sup_ims, sup_labels)
            print('Done.')
            end1 = time.time()

            corrects = 0
            total = 0
            if 0:
                test_ims, test_labels = mnist_data['test_image'], mnist_data['test_label']
            else:
                test_ims = ims2k
                test_labels = label2k


            ims_batches = np.array_split(test_ims, 100)
            labels_batches = np.array_split(test_labels, 100)

            def format_error_rate(pr):
                return "{:.2f}%".format(100*(1-pr))

            start2 = time.time()
            args = (tup+(clnet,) for tup in itr.izip(ims_batches, labels_batches))
            for i, res in enumerate(pnet.parallel.starmap(test, args)):
                corrects += res.sum()
                total += res.size
                pr = corrects / total
            end2 = time.time()

            error_rate = 1.0 - pr
            error_rates.append(error_rate)
            print('error rate', error_rate * 100)

            sup_training_times.append(end1 - start1)
            testing_times.append(end2 - start2)

            print('times', end1-start1, end2-start2)
