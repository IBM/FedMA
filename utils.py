import os
import argparse
import json
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import logging
import torchvision
import torchvision.transforms as transforms
import torch.utils.data as data
from itertools import product
import math
import copy
import time
from sklearn.metrics import confusion_matrix

# we've changed to a faster solver
#from scipy.optimize import linear_sum_assignment
import logging

from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import torch.nn.init as init
from datasets import MNIST_truncated, CIFAR10_truncated, ImageFolderTruncated, CIFAR10ColorGrayScaleTruncated
from combine_nets import prepare_uniform_weights, prepare_sanity_weights, prepare_weight_matrix, normalize_weights, get_weighted_average_pred

from vgg import *
from model import *

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

def mkdirs(dirpath):
    try:
        os.makedirs(dirpath)
    except Exception as _:
        pass

def parse_class_dist(net_class_config):

    cls_net_map = {}

    for net_idx, net_classes in enumerate(net_class_config):
        for net_cls in net_classes:
            if net_cls not in cls_net_map:
                cls_net_map[net_cls] = []
            cls_net_map[net_cls].append(net_idx)

    return cls_net_map

def record_net_data_stats(y_train, net_dataidx_map, logdir):

    net_cls_counts = {}

    for net_i, dataidx in net_dataidx_map.items():
        unq, unq_cnt = np.unique(y_train[dataidx], return_counts=True)
        tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
        net_cls_counts[net_i] = tmp
    logging.debug('Data statistics: %s' % str(net_cls_counts))
    return net_cls_counts

def partition_data(dataset, datadir, logdir, partition, n_nets, alpha, args):

    if dataset == 'mnist':
        X_train, y_train, X_test, y_test = load_mnist_data(datadir)
        n_train = X_train.shape[0]
    elif dataset == 'cifar10':
        X_train, y_train, X_test, y_test = load_cifar10_data(datadir)
        n_train = X_train.shape[0]
    elif dataset == 'cinic10':
        _train_dir = './data/cinic10/cinic-10-trainlarge/train'
        cinic_mean = [0.47889522, 0.47227842, 0.43047404]
        cinic_std = [0.24205776, 0.23828046, 0.25874835]
        trainset = ImageFolderTruncated(_train_dir, transform=transforms.Compose([transforms.ToTensor(),
                                                            transforms.Lambda(lambda x: F.pad(Variable(x.unsqueeze(0), 
                                                                                            requires_grad=False),
                                                                                            (4,4,4,4),mode='reflect').data.squeeze()),
                                                            transforms.ToPILImage(),
                                                            transforms.RandomCrop(32),
                                                            transforms.RandomHorizontalFlip(),
                                                            transforms.ToTensor(),
                                                            transforms.Normalize(mean=cinic_mean,std=cinic_std),
                                                            ]))
        y_train = trainset.get_train_labels
        n_train = y_train.shape[0]

    if partition == "homo":
        idxs = np.random.permutation(n_train)
        batch_idxs = np.array_split(idxs, n_nets)
        net_dataidx_map = {i: batch_idxs[i] for i in range(n_nets)}

    elif partition == "hetero-dir":
        min_size = 0
        K = 10
        N = y_train.shape[0]
        net_dataidx_map = {}

        while min_size < 10:
            idx_batch = [[] for _ in range(n_nets)]
            # for each class in the dataset
            for k in range(K):
                idx_k = np.where(y_train == k)[0]
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(np.repeat(alpha, n_nets))
                ## Balance
                proportions = np.array([p*(len(idx_j)<N/n_nets) for p,idx_j in zip(proportions,idx_batch)])
                proportions = proportions/proportions.sum()
                proportions = (np.cumsum(proportions)*len(idx_k)).astype(int)[:-1]
                idx_batch = [idx_j + idx.tolist() for idx_j,idx in zip(idx_batch,np.split(idx_k,proportions))]
                min_size = min([len(idx_j) for idx_j in idx_batch])

        for j in range(n_nets):
            np.random.shuffle(idx_batch[j])
            net_dataidx_map[j] = idx_batch[j]

    elif partition == "hetero-fbs":
        # in this part we conduct a experimental study on exploring the effect of increasing the number of batches
        # but the number of data points are approximately fixed for each batch
        # the arguments we need to use here are: `args.partition_step_size`, `args.local_points`, `args.partition_step`(step can be {0, 1, ..., args.partition_step_size - 1}).
        # Note that `args.partition` need to be fixed as "hetero-fbs" where fbs means fixed batch size
        net_dataidx_map = {}

        # stage 1st: homo partition
        idxs = np.random.permutation(n_train)
        total_num_batches = int(n_train/args.local_points) # e.g. currently we have 180k, we want each local batch has 5k data points the `total_num_batches` becomes 36
        step_batch_idxs = np.array_split(idxs, args.partition_step_size)
        
        sub_partition_size = int(total_num_batches / args.partition_step_size) # e.g. for `total_num_batches` at 36 and `args.partition_step_size` at 6, we have `sub_partition_size` at 6

        # stage 2nd: hetero partition
        n_batches = (args.partition_step + 1) * sub_partition_size
        min_size = 0
        K = 10

        #N = len(step_batch_idxs[args.step])
        baseline_indices = np.concatenate([step_batch_idxs[i] for i in range(args.partition_step + 1)])
        y_train = y_train[baseline_indices]
        N = y_train.shape[0]

        while min_size < 10:
            idx_batch = [[] for _ in range(n_batches)]
            # for each class in the dataset
            for k in range(K):
                idx_k = np.where(y_train == k)[0]
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(np.repeat(alpha, n_batches))
                ## Balance
                proportions = np.array([p*(len(idx_j)<N/n_batches) for p,idx_j in zip(proportions,idx_batch)])
                proportions = proportions/proportions.sum()
                proportions = (np.cumsum(proportions)*len(idx_k)).astype(int)[:-1]
                idx_batch = [idx_j + idx.tolist() for idx_j,idx in zip(idx_batch,np.split(idx_k,proportions))]
                min_size = min([len(idx_j) for idx_j in idx_batch])

        # we leave this to the end
        for j in range(n_batches):
            np.random.shuffle(idx_batch[j])
            net_dataidx_map[j] = idx_batch[j]

        traindata_cls_counts = record_net_data_stats(y_train, net_dataidx_map, logdir)
        return y_train, net_dataidx_map, traindata_cls_counts, baseline_indices

    traindata_cls_counts = record_net_data_stats(y_train, net_dataidx_map, logdir)

    #return (X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts)
    return y_train, net_dataidx_map, traindata_cls_counts


def partition_data_dist_skew(dataset, datadir, logdir, partition, n_nets, alpha, args):
    if dataset == 'mnist':
        pass
    elif dataset == 'cifar10':
        normalize = transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
                                std=[x/255.0 for x in [63.0, 62.1, 66.7]])
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: F.pad(
                                Variable(x.unsqueeze(0), requires_grad=False),
                                (4,4,4,4),mode='reflect').data.squeeze()),
            transforms.ToPILImage(),
            transforms.RandomCrop(32),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
            ])
        # data prep for test set
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            normalize])
        # load training and test set here:
        training_set = torchvision.datasets.CIFAR10(root='./data/cifar10', train=True,
                                                download=True, transform=None)

        testset = torchvision.datasets.CIFAR10(root='./data/cifar10', train=False,
                                               download=True, transform=None)

        y_train = np.array(copy.deepcopy(training_set.targets))
        n_train = training_set.data.shape[0]

        entire_gray_scale_indices_train = []
        entire_gray_scale_indices_test = []

        # we start an adjust version here:
        #########################################################################################################################################
        # in this setting, we corelate the majority / minory with the class
        # i.e. we firstly do an extreme version where we randomly sample 5 out of 10 groups s.t. in those groups there are only grayscale images
        #      for the other five groups, we leave all images to be colored images
        #########################################################################################################################################
        grayscale_dominate_classes = np.random.choice(np.arange(10), 5, replace=False)
        logger.info("Grayscale image dominated classes are : {}".format(grayscale_dominate_classes))

        # we split all grayscale dominate classes to client 0 and all color dominate classes to client1
        client0_indices = []
        client1_indices = []
        for i in range(10):
            if i in grayscale_dominate_classes:
                logger.info("Grayscale dominate class index: {}".format(i))
                class_indices_train = np.where(np.array(training_set.targets) == i)[0]
                # we fix this to be one first

                ###
                # this is the extreme case, we now change to a relatexed case
                ###
                #num_of_gray_scale_per_class_train = int(1.0 * class_indices_train.shape[0])
                num_of_gray_scale_per_class_train = int(0.95 * class_indices_train.shape[0])
                class_gray_scale_indices_train = np.random.choice(class_indices_train, num_of_gray_scale_per_class_train, replace=False)
                client0_indices.append(class_indices_train)
            else:
                logger.info("Color dominate class index: {}".format(i))
                class_indices_train = np.where(np.array(training_set.targets) == i)[0]
                num_of_gray_scale_per_class_train = int(0.05 * class_indices_train.shape[0])
                class_gray_scale_indices_train = np.random.choice(class_indices_train, num_of_gray_scale_per_class_train, replace=False)
                client1_indices.append(class_indices_train)
            entire_gray_scale_indices_train.append(class_gray_scale_indices_train)
        entire_gray_scale_indices_train = np.concatenate(entire_gray_scale_indices_train)

        client0_indices = np.concatenate(client0_indices)
        client1_indices = np.concatenate(client1_indices)

        ###
        # extreme case:
        ###
        for i in range(10):
            class_indices_test = np.where(np.array(testset.targets) == i)[0]
            # training set contains skewness, but in test set colored and gray-scale images are evenly distributed
            num_of_gray_scale_per_class_test = int(0.5 * class_indices_test.shape[0])
            class_gray_scale_indices_test = np.random.choice(class_indices_test, num_of_gray_scale_per_class_test, replace=False)
            entire_gray_scale_indices_test.append(class_gray_scale_indices_test)
            logger.info("Num of gray scale image per class test: {}".format(class_gray_scale_indices_test.shape[0]))
        entire_gray_scale_indices_test = np.concatenate(entire_gray_scale_indices_test)
        logger.info("Total Num of gray scale image test: {}".format(entire_gray_scale_indices_test.shape[0]))


    elif dataset == 'cinic10':
        pass

    if partition == "homo":
        net_dataidx_map = {}
        idxs = np.arange(n_train)

        indices_colored = np.array([i for i in idxs if i not in entire_gray_scale_indices_train])

        # we split grayscale and colored images on two workers entirely
        net_dataidx_map[0] = client0_indices
        net_dataidx_map[1] = client1_indices

    elif partition == "hetero-dir":
        pass
    elif partition == "hetero-fbs":
        pass

    traindata_cls_counts = record_net_data_stats(y_train, net_dataidx_map, logdir)
    return y_train, net_dataidx_map, traindata_cls_counts, entire_gray_scale_indices_train, entire_gray_scale_indices_test


def partition_data_dist_skew_baseline(dataset, datadir, logdir, partition, n_nets, alpha, args):
    '''
    This is for one of the baseline we're going to use for rebuttal of ICLR2020
    Where the entire training dataset and the entire test dataset are with all grayscale images
    '''
    if dataset == 'mnist':
        pass
    elif dataset == 'cifar10':
        normalize = transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
                                std=[x/255.0 for x in [63.0, 62.1, 66.7]])
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: F.pad(
                                Variable(x.unsqueeze(0), requires_grad=False),
                                (4,4,4,4),mode='reflect').data.squeeze()),
            transforms.ToPILImage(),
            transforms.RandomCrop(32),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
            ])
        # data prep for test set
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            normalize])
        # load training and test set here:
        training_set = torchvision.datasets.CIFAR10(root='./data/cifar10', train=True,
                                                download=True, transform=None)

        testset = torchvision.datasets.CIFAR10(root='./data/cifar10', train=False,
                                               download=True, transform=None)

        y_train = np.array(copy.deepcopy(training_set.targets))
        n_train = training_set.data.shape[0]
        n_test = testset.data.shape[0]

        entire_gray_scale_indices_train = np.arange(n_train)
        entire_gray_scale_indices_test = np.arange(n_test)

    elif dataset == 'cinic10':
        pass

    if partition == "homo":
        net_dataidx_map = {}
        idxs = np.arange(n_train)

        indices_colored = np.array([i for i in idxs if i not in entire_gray_scale_indices_train])

        # we split grayscale and colored images on two workers entirely
        net_dataidx_map[0] = np.arange(n_train)

    elif partition == "hetero-dir":
        pass
    elif partition == "hetero-fbs":
        pass

    traindata_cls_counts = record_net_data_stats(y_train, net_dataidx_map, logdir)
    return y_train, net_dataidx_map, traindata_cls_counts, entire_gray_scale_indices_train, entire_gray_scale_indices_test
    #return y_train, entire_gray_scale_indices_train, entire_gray_scale_indices_test


def partition_data_dist_skew_normal(dataset, datadir, logdir, partition, n_nets, alpha, args):
    '''
    This is for one of the baseline we're going to use for rebuttal of ICLR2020
    Where the entire training dataset and the entire test dataset are with all grayscale images
    '''
    if dataset == 'mnist':
        pass
    elif dataset == 'cifar10':
        normalize = transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
                                std=[x/255.0 for x in [63.0, 62.1, 66.7]])
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: F.pad(
                                Variable(x.unsqueeze(0), requires_grad=False),
                                (4,4,4,4),mode='reflect').data.squeeze()),
            transforms.ToPILImage(),
            transforms.RandomCrop(32),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
            ])
        # data prep for test set
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            normalize])
        # load training and test set here:
        training_set = torchvision.datasets.CIFAR10(root='./data/cifar10', train=True,
                                                download=True, transform=None)

        testset = torchvision.datasets.CIFAR10(root='./data/cifar10', train=False,
                                               download=True, transform=None)

        y_train = np.array(copy.deepcopy(training_set.targets))
        n_train = training_set.data.shape[0]
        n_test = testset.data.shape[0]

        entire_gray_scale_indices_train = []
        entire_gray_scale_indices_test = np.arange(n_test)

    elif dataset == 'cinic10':
        pass

    if partition == "homo":
        net_dataidx_map = {}
        idxs = np.arange(n_train)

        indices_colored = np.array([i for i in idxs if i not in entire_gray_scale_indices_train])

        # we split grayscale and colored images on two workers entirely
        net_dataidx_map[0] = np.arange(n_train)

    elif partition == "hetero-dir":
        pass
    elif partition == "hetero-fbs":
        pass

    traindata_cls_counts = record_net_data_stats(y_train, net_dataidx_map, logdir)
    return y_train, net_dataidx_map, traindata_cls_counts, entire_gray_scale_indices_train, entire_gray_scale_indices_test


def partition_data_dist_skew_baseline_balanced(dataset, datadir, logdir, partition, n_nets, alpha, args):
    if dataset == 'mnist':
        pass
    elif dataset == 'cifar10':
        normalize = transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
                                std=[x/255.0 for x in [63.0, 62.1, 66.7]])
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: F.pad(
                                Variable(x.unsqueeze(0), requires_grad=False),
                                (4,4,4,4),mode='reflect').data.squeeze()),
            transforms.ToPILImage(),
            transforms.RandomCrop(32),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
            ])
        # data prep for test set
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            normalize])
        # load training and test set here:
        training_set = torchvision.datasets.CIFAR10(root='./data/cifar10', train=True,
                                                download=True, transform=None)

        testset = torchvision.datasets.CIFAR10(root='./data/cifar10', train=False,
                                               download=True, transform=None)

        y_train = np.array(copy.deepcopy(training_set.targets))
        n_train = training_set.data.shape[0]

        entire_gray_scale_indices_train = []
        entire_gray_scale_indices_test = []

        # we start an adjust version here:
        #########################################################################################################################################
        # in this setting, we corelate the majority / minory with the class
        # i.e. we firstly do an extreme version where we randomly sample 5 out of 10 groups s.t. in those groups there are only grayscale images
        #      for the other five groups, we leave all images to be colored images
        #########################################################################################################################################

        grayscale_dominate_classes = np.random.choice(np.arange(10), 5, replace=False)
        logger.info("Grayscale image dominated classes are : {}".format(grayscale_dominate_classes))

        # we split all grayscale dominate classes to client 0 and all color dominate classes to client1
        for i in range(10):
            if i in grayscale_dominate_classes:
                logger.info("Grayscale dominate class index: {}".format(i))
                class_indices_train = np.where(np.array(training_set.targets) == i)[0]
                # we fix this to be one first

                ###
                # this is the extreme case, we now change to a relatexed case
                ###
                #num_of_gray_scale_per_class_train = int(1.0 * class_indices_train.shape[0])
                num_of_gray_scale_per_class_train = int(0.5 * class_indices_train.shape[0])
                class_gray_scale_indices_train = np.random.choice(class_indices_train, num_of_gray_scale_per_class_train, replace=False)
            else:
                logger.info("Color dominate class index: {}".format(i))
                class_indices_train = np.where(np.array(training_set.targets) == i)[0]
                num_of_gray_scale_per_class_train = int(0.5 * class_indices_train.shape[0])
                class_gray_scale_indices_train = np.random.choice(class_indices_train, num_of_gray_scale_per_class_train, replace=False)
            entire_gray_scale_indices_train.append(class_gray_scale_indices_train)
        entire_gray_scale_indices_train = np.concatenate(entire_gray_scale_indices_train)

        ####################
        # in this test case
        ####################
        for i in range(10):
            class_indices_test = np.where(np.array(testset.targets) == i)[0]
            # training set contains skewness, but in test set colored and gray-scale images are evenly distributed
            num_of_gray_scale_per_class_test = int(0.5 * class_indices_test.shape[0])
            class_gray_scale_indices_test = np.random.choice(class_indices_test, num_of_gray_scale_per_class_test, replace=False)
            entire_gray_scale_indices_test.append(class_gray_scale_indices_test)
            logger.info("Num of gray scale image per class test: {}".format(class_gray_scale_indices_test.shape[0]))
        entire_gray_scale_indices_test = np.concatenate(entire_gray_scale_indices_test)
        logger.info("Total Num of gray scale image test: {}".format(entire_gray_scale_indices_test.shape[0]))


    elif dataset == 'cinic10':
        pass

    if partition == "homo":
        net_dataidx_map = {}
        idxs = np.arange(n_train)

        net_dataidx_map[0] = np.arange(n_train)

    elif partition == "hetero-dir":
        pass
    elif partition == "hetero-fbs":
        pass

    traindata_cls_counts = record_net_data_stats(y_train, net_dataidx_map, logdir)
    return y_train, net_dataidx_map, traindata_cls_counts, entire_gray_scale_indices_train, entire_gray_scale_indices_test
    #return y_train, entire_gray_scale_indices_train, entire_gray_scale_indices_test


def partition_data_dist_skew_baseline_oversampled(dataset, datadir, logdir, partition, n_nets, alpha, args):

    if dataset == 'mnist':
        pass
    elif dataset == 'cifar10':
        normalize = transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
                                std=[x/255.0 for x in [63.0, 62.1, 66.7]])
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: F.pad(
                                Variable(x.unsqueeze(0), requires_grad=False),
                                (4,4,4,4),mode='reflect').data.squeeze()),
            transforms.ToPILImage(),
            transforms.RandomCrop(32),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
            ])
        # data prep for test set
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            normalize])
        # load training and test set here:
        training_set = torchvision.datasets.CIFAR10(root='./data/cifar10', train=True,
                                                download=True, transform=None)

        testset = torchvision.datasets.CIFAR10(root='./data/cifar10', train=False,
                                               download=True, transform=None)

        y_train = np.array(copy.deepcopy(training_set.targets))
        n_train = training_set.data.shape[0]

        entire_gray_scale_indices_train = []
        entire_gray_scale_indices_test = []


        # we start an adjust version here:
        #########################################################################################################################################
        # in this setting, we corelate the majority / minory with the class
        # i.e. we firstly do an extreme version where we randomly sample 5 out of 10 groups s.t. in those groups there are only grayscale images
        #      for the other five groups, we leave all images to be colored images
        #########################################################################################################################################
        grayscale_dominate_classes = np.random.choice(np.arange(10), 5, replace=False)
        logger.info("Grayscale image dominated classes are : {}".format(grayscale_dominate_classes))

        # we split all grayscale dominate classes to client 0 and all color dominate classes to client1
        entire_indices = []

        for i in range(10):
            if i in grayscale_dominate_classes:
                logger.info("Grayscale dominate class index: {}".format(i))
                class_indices_train = np.where(np.array(training_set.targets) == i)[0]
                # we fix this to be one first

                ###
                # this is the extreme case, we now change to a relatexed case
                ###
                # here we created the dominated images
                num_of_gray_scale_per_class_train = int(0.95 * class_indices_train.shape[0])
                class_gray_scale_indices_train = np.random.choice(class_indices_train, num_of_gray_scale_per_class_train, replace=False)
                entire_indices.append(class_indices_train)
                # here we need to oversample the underrepresented images
                class_color_indices_train = [idx for idx in class_indices_train if idx not in class_gray_scale_indices_train]
                
                logger.info("Length gray scale image train: {}".format(len(class_gray_scale_indices_train)))
                logger.info("Length gray scale image test: {}".format(len(class_color_indices_train)))
                # we use this way to mitigate the data bias by oversampling
                for i in range(int(0.95/0.05)):
                    entire_indices.append(class_color_indices_train)
            else:
                logger.info("Color dominate class index: {}".format(i))
                class_indices_train = np.where(np.array(training_set.targets) == i)[0]
                num_of_gray_scale_per_class_train = int(0.05 * class_indices_train.shape[0])
                class_gray_scale_indices_train = np.random.choice(class_indices_train, num_of_gray_scale_per_class_train, replace=False)
                
                logger.info("Length gray scale image train: {}".format(len(class_gray_scale_indices_train)))
                entire_indices.append(class_indices_train)

                # we use this way to mitigate the data bias by oversampling
                for i in range(int(0.95/0.05)):
                    entire_indices.append(class_gray_scale_indices_train)

            entire_gray_scale_indices_train.append(class_gray_scale_indices_train)
        entire_gray_scale_indices_train = np.concatenate(entire_gray_scale_indices_train)

        entire_indices = np.concatenate(entire_indices)
        logger.info("Entire indices: {}".format(len(entire_indices)))

        ###
        # extreme case:
        ###
        for i in range(10):
            class_indices_test = np.where(np.array(testset.targets) == i)[0]
            # training set contains skewness, but in test set colored and gray-scale images are evenly distributed
            num_of_gray_scale_per_class_test = int(0.5 * class_indices_test.shape[0])
            class_gray_scale_indices_test = np.random.choice(class_indices_test, num_of_gray_scale_per_class_test, replace=False)
            entire_gray_scale_indices_test.append(class_gray_scale_indices_test)
            logger.info("Num of gray scale image per class test: {}".format(class_gray_scale_indices_test.shape[0]))
        entire_gray_scale_indices_test = np.concatenate(entire_gray_scale_indices_test)
        logger.info("Total Num of gray scale image test: {}".format(entire_gray_scale_indices_test.shape[0]))


    elif dataset == 'cinic10':
        pass

    if partition == "homo":
        net_dataidx_map = {}
        idxs = np.arange(n_train)

        indices_colored = np.array([i for i in idxs if i not in entire_gray_scale_indices_train])
        # we split grayscale and colored images on two workers entirely
        net_dataidx_map[0] = entire_indices

    elif partition == "hetero-dir":
        pass
    elif partition == "hetero-fbs":
        pass

    traindata_cls_counts = record_net_data_stats(y_train, net_dataidx_map, logdir)
    return y_train, net_dataidx_map, traindata_cls_counts, entire_gray_scale_indices_train, entire_gray_scale_indices_test
    #return y_train, net_dataidx_map, entire_gray_scale_indices_train, entire_gray_scale_indices_test



def partition_data_viz(dataset, datadir, logdir, partition, n_nets, alpha, args):
    if dataset == 'mnist':
        pass
    elif dataset == 'cifar10':
        # load training and test set here:
        training_set = torchvision.datasets.CIFAR10(root='./data/cifar10', train=True,
                                                download=True, transform=None)

        testset = torchvision.datasets.CIFAR10(root='./data/cifar10', train=False,
                                               download=True, transform=None)

        y_train = np.array(copy.deepcopy(training_set.targets))
        n_train = training_set.data.shape[0]

        # we start an adjust version here:
        ###############################################################################################
        # Our strategy is like the following:
        # we split the data of CIFAR=10 dataset to two different clients
        # and those two clients only share on common class
        ###############################################################################################
        # the class 5 (i.e. dog) is shared across the two clients
        #classes_client1 = [class_index for class_index in range(0, 7)]
        #classes_client2 = [class_index for class_index in range(2, 10)]
        #classes_client1 = [0, 1]
        #classes_client2 = [8, 9]

        classes_client1 = [0, 1, 2, 3]
        classes_client2 = [6, 7, 8, 9]

        # we split all grayscale dominate classes to client 0 and all color dominate classes to client1
        client0_indices = []
        client1_indices = []
        for ci in range(10):
            class_indices_train = np.where(np.array(training_set.targets) == ci)[0]
            #logger.info("############# class index: {}, class_indices: {}".format(ci, class_indices_train))
            if ci in classes_client1:
                logger.info("Client 1 exclusive classes: {}".format(ci))
                client0_indices.append(class_indices_train)
            elif ci in classes_client2:
                logger.info("Client 2 exclusive classes: {}".format(ci))
                
                client1_indices.append(class_indices_train)
            else:
                # here we handel the shared class
                num_of_dp_per_client = int(0.5 * class_indices_train.shape[0])
                shared_class_indices_client0 = np.random.choice(class_indices_train, num_of_dp_per_client, replace=False)
                shared_class_indices_client1 = [idx for idx in class_indices_train if idx not in shared_class_indices_client0]
                client0_indices.append(shared_class_indices_client0)
                client1_indices.append(shared_class_indices_client1)
                logger.info("shared_class_indices_client0: {}, length: {}, shared_class_indices_client1: {}, length: {}".format(
                    shared_class_indices_client0, len(shared_class_indices_client0), shared_class_indices_client1, len(shared_class_indices_client1)))

        client0_indices = np.concatenate(client0_indices)
        client1_indices = np.concatenate(client1_indices)

    elif dataset == 'cinic10':
        pass

    if partition == "homo":
        net_dataidx_map = {}
        idxs = np.arange(n_train)

        # we split grayscale and colored images on two workers entirely
        net_dataidx_map[0] = client0_indices
        net_dataidx_map[1] = client1_indices

    elif partition == "hetero-dir":
        pass
    elif partition == "hetero-fbs":
        pass

    traindata_cls_counts = record_net_data_stats(y_train, net_dataidx_map, logdir)
    return y_train, net_dataidx_map, traindata_cls_counts


def partition_data_viz2(dataset, datadir, logdir, partition, n_nets, alpha, args):
    if dataset == 'mnist':
        pass
    elif dataset == 'cifar10':
        normalize = transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
                                std=[x/255.0 for x in [63.0, 62.1, 66.7]])
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: F.pad(
                                Variable(x.unsqueeze(0), requires_grad=False),
                                (4,4,4,4),mode='reflect').data.squeeze()),
            transforms.ToPILImage(),
            transforms.RandomCrop(32),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
            ])
        # data prep for test set
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            normalize])
        # load training and test set here:
        training_set = torchvision.datasets.CIFAR10(root='./data/cifar10', train=True,
                                                download=True, transform=None)

        testset = torchvision.datasets.CIFAR10(root='./data/cifar10', train=False,
                                               download=True, transform=None)

        y_train = np.array(copy.deepcopy(training_set.targets))
        n_train = training_set.data.shape[0]

        entire_gray_scale_indices_train = []
        entire_gray_scale_indices_test = []

        # we start an adjust version here:
        ###############################################################################################
        # in this setting, we corelate the majority / minory with the class
        # i.e. we firstly do an extreme version where we randomly sample 5 out of 10 groups 
        #      s.t. in those groups there are only grayscale images
        #      for the other five groups, we leave all images to be colored images
        ###############################################################################################
        #grayscale_dominate_classes = np.random.choice(np.arange(10), 5, replace=False)
        grayscale_dominate_classes = np.arange(0, 6)
        logger.info("Grayscale image dominated classes are : {}".format(grayscale_dominate_classes))

        # we split all grayscale dominate classes to client 0 and all color dominate classes to client1
        client0_indices = []
        client1_indices = []
        for i in range(10):
            if i in grayscale_dominate_classes:
                logger.info("Grayscale dominate class index: {}".format(i))
                class_indices_train = np.where(np.array(training_set.targets) == i)[0]
                # we fix this to be one first

                ###
                # this is the extreme case, we now change to a relatexed case
                ###
                num_of_gray_scale_per_class_train = int(1.0 * class_indices_train.shape[0])
                #num_of_gray_scale_per_class_train = int(0.95 * class_indices_train.shape[0])
                class_gray_scale_indices_train = np.random.choice(class_indices_train, num_of_gray_scale_per_class_train, replace=False)
                client0_indices.append(class_indices_train)

                entire_gray_scale_indices_train.append(class_gray_scale_indices_train)
            else:
                logger.info("Color dominate class index: {}".format(i))
                class_indices_train = np.where(np.array(training_set.targets) == i)[0]
                #num_of_gray_scale_per_class_train = int(0.05 * class_indices_train.shape[0])
                num_of_gray_scale_per_class_train = int(0.0 * class_indices_train.shape[0])
                #class_gray_scale_indices_train = np.random.choice(class_indices_train, num_of_gray_scale_per_class_train, replace=False)
                client1_indices.append(class_indices_train)
            #entire_gray_scale_indices_train.append(class_gray_scale_indices_train)
        entire_gray_scale_indices_train = np.concatenate(entire_gray_scale_indices_train)

        client0_indices = np.concatenate(client0_indices)
        client1_indices = np.concatenate(client1_indices)

        ###
        # extreme case:
        ###
        for i in range(10):
            class_indices_test = np.where(np.array(testset.targets) == i)[0]
            # training set contains skewness, but in test set colored and gray-scale images are evenly distributed
            num_of_gray_scale_per_class_test = int(0.5 * class_indices_test.shape[0])
            class_gray_scale_indices_test = np.random.choice(class_indices_test, num_of_gray_scale_per_class_test, replace=False)
            entire_gray_scale_indices_test.append(class_gray_scale_indices_test)
            logger.info("Num of gray scale image per class test: {}".format(class_gray_scale_indices_test.shape[0]))
        entire_gray_scale_indices_test = np.concatenate(entire_gray_scale_indices_test)
        logger.info("Total Num of gray scale image test: {}".format(entire_gray_scale_indices_test.shape[0]))


    elif dataset == 'cinic10':
        pass

    if partition == "homo":
        net_dataidx_map = {}
        idxs = np.arange(n_train)
        indices_colored = np.array([i for i in idxs if i not in entire_gray_scale_indices_train])

        # we split grayscale and colored images on two workers entirely
        net_dataidx_map[0] = client0_indices
        net_dataidx_map[1] = client1_indices

    elif partition == "hetero-dir":
        pass
    elif partition == "hetero-fbs":
        pass

    traindata_cls_counts = record_net_data_stats(y_train, net_dataidx_map, logdir)
    return y_train, net_dataidx_map, traindata_cls_counts, entire_gray_scale_indices_train, entire_gray_scale_indices_test



def load_mnist_data(datadir):
    transform = transforms.Compose([transforms.ToTensor()])

    mnist_train_ds = MNIST_truncated(datadir, train=True, download=True, transform=transform)
    mnist_test_ds = MNIST_truncated(datadir, train=False, download=True, transform=transform)

    X_train, y_train = mnist_train_ds.data, mnist_train_ds.target
    X_test, y_test = mnist_test_ds.data, mnist_test_ds.target

    X_train = X_train.data.numpy()
    y_train = y_train.data.numpy()
    X_test = X_test.data.numpy()
    y_test = y_test.data.numpy()

    return (X_train, y_train, X_test, y_test)

def load_cifar10_data(datadir):

    transform = transforms.Compose([transforms.ToTensor()])

    cifar10_train_ds = CIFAR10_truncated(datadir, train=True, download=True, transform=transform)
    cifar10_test_ds = CIFAR10_truncated(datadir, train=False, download=True, transform=transform)

    X_train, y_train = cifar10_train_ds.data, cifar10_train_ds.target
    X_test, y_test = cifar10_test_ds.data, cifar10_test_ds.target

    return (X_train, y_train, X_test, y_test)

def compute_accuracy(model, dataloader, get_confusion_matrix=False, device="cpu"):

    was_training = False
    if model.training:
        model.eval()
        was_training = True

    true_labels_list, pred_labels_list = np.array([]), np.array([])

    correct, total = 0, 0
    with torch.no_grad():
        for batch_idx, (x, target) in enumerate(dataloader):
            x, target = x.to(device), target.to(device)
            out = model(x)
            _, pred_label = torch.max(out.data, 1)

            total += x.data.size()[0]
            correct += (pred_label == target.data).sum().item()

            if device == "cpu":
                pred_labels_list = np.append(pred_labels_list, pred_label.numpy())
                true_labels_list = np.append(true_labels_list, target.data.numpy())
            else:
                pred_labels_list = np.append(pred_labels_list, pred_label.cpu().numpy())
                true_labels_list = np.append(true_labels_list, target.data.cpu().numpy())               

    if get_confusion_matrix:
        conf_matrix = confusion_matrix(true_labels_list, pred_labels_list)

    if was_training:
        model.train()

    if get_confusion_matrix:
        return correct/float(total), conf_matrix

    return correct/float(total)


def init_cnns(net_configs, n_nets):
    '''
    Initialize the local CNNs
    Please note that this part is hard coded right now
    '''

    input_size = (16 * 5 * 5) # hard coded, defined by the SimpleCNN useds
    output_size = net_configs[-1] #
    hidden_sizes = [120, 84]

    cnns = {net_i: None for net_i in range(n_nets)}

    # we add this book keeping to store meta data of model weights
    model_meta_data = []
    layer_type = []

    for cnn_i in range(n_nets):
        cnn = SimpleCNN(input_size, hidden_sizes, output_size)

        cnns[cnn_i] = cnn

    for (k, v) in cnns[0].state_dict().items():
        model_meta_data.append(v.shape)
        layer_type.append(k)
        #logger.info("Layer name: {}, layer shape: {}".format(k, v.shape))
    return cnns, model_meta_data, layer_type


def init_models(net_configs, n_nets, args):
    '''
    Initialize the local LeNets
    Please note that this part is hard coded right now
    '''

    cnns = {net_i: None for net_i in range(n_nets)}

    # we add this book keeping to store meta data of model weights
    model_meta_data = []
    layer_type = []

    for cnn_i in range(n_nets):
        if args.model == "lenet":
            cnn = LeNet()
        elif args.model == "vgg":
            cnn = vgg11()
        elif args.model == "simple-cnn":
            if args.dataset in ("cifar10", "cinic10"):
                cnn = SimpleCNN(input_dim=(16 * 5 * 5), hidden_dims=[120, 84], output_dim=10)
            elif args.dataset == "mnist":
                cnn = SimpleCNNMNIST(input_dim=(16 * 4 * 4), hidden_dims=[120, 84], output_dim=10)
        elif args.model == "moderate-cnn":
            if args.dataset == "mnist":
                cnn = ModerateCNNMNIST()
            elif args.dataset in ("cifar10", "cinic10"):
                cnn = ModerateCNN()

        cnns[cnn_i] = cnn

    for (k, v) in cnns[0].state_dict().items():
        model_meta_data.append(v.shape)
        layer_type.append(k)
        #logger.info("{} ::: Layer name: {}, layer shape: {}".format(args.model, k, v.shape))
    return cnns, model_meta_data, layer_type


def save_model(model, model_index):
    logger.info("saving local model-{}".format(model_index))
    with open("trained_local_model"+str(model_index), "wb") as f_:
        torch.save(model.state_dict(), f_)
    return

def load_model(model, model_index, rank=0, device="cpu"):
    #
    with open("trained_local_model"+str(model_index), "rb") as f_:
        model.load_state_dict(torch.load(f_))
    model.to(device)
    return model


def save_model_dist_skew(model, model_index):
    logger.info("saving local model-{} dist skew".format(model_index))
    with open("trained_local_model_dist_skew"+str(model_index), "wb") as f_:
        torch.save(model.state_dict(), f_)
    return

def load_model_dist_skew(model, model_index, device="cpu"):
    logger.info("loading local model-{} dist skew".format(model_index))
    with open("trained_local_model_dist_skew"+str(model_index), "rb") as f_:
        model.load_state_dict(torch.load(f_))
    model.to(device)
    return model


def save_model_viz(model, model_index):
    logger.info("saving local model-{} visulization".format(model_index))
    with open("trained_local_model_viz{}_new".format(model_index), "wb") as f_:
        torch.save(model.state_dict(), f_)
    return

def load_model_viz(model, model_index, device="cpu"):
    logger.info("loading local model-{} visulization".format(model_index))
    with open("trained_local_model_viz"+str(model_index), "rb") as f_:
        model.load_state_dict(torch.load(f_))
    model.to(device)
    return model


def get_dataloader(dataset, datadir, train_bs, test_bs, dataidxs=None):

    if dataset in ('mnist', 'cifar10'):
        if dataset == 'mnist':
            dl_obj = MNIST_truncated

            transform_train = transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))])

            transform_test = transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))])

        elif dataset == 'cifar10':
            dl_obj = CIFAR10_truncated

            normalize = transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
                                std=[x/255.0 for x in [63.0, 62.1, 66.7]])
            transform_train = transforms.Compose([
                transforms.ToTensor(),
                transforms.Lambda(lambda x: F.pad(
                                    Variable(x.unsqueeze(0), requires_grad=False),
                                    (4,4,4,4),mode='reflect').data.squeeze()),
                transforms.ToPILImage(),
                transforms.RandomCrop(32),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
                ])
            # data prep for test set
            transform_test = transforms.Compose([transforms.ToTensor(),normalize])


        train_ds = dl_obj(datadir, dataidxs=dataidxs, train=True, transform=transform_train, download=True)
        test_ds = dl_obj(datadir, train=False, transform=transform_test, download=True)

        train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, shuffle=True)
        test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False)

    elif dataset == 'cinic10':
        # statistic for normalizing the dataset
        cinic_mean = [0.47889522, 0.47227842, 0.43047404]
        cinic_std = [0.24205776, 0.23828046, 0.25874835]

        cinic_directory = './data/cinic10'

        training_set = ImageFolderTruncated(cinic_directory + '/cinic-10-trainlarge/train', 
                                                                        dataidxs=dataidxs,
                                                                        transform=transforms.Compose([transforms.ToTensor(),
                                                                                                     transforms.Lambda(lambda x: F.pad(Variable(x.unsqueeze(0), requires_grad=False),
                                                                                                     (4,4,4,4),mode='reflect').data.squeeze()),
                                                                                                     transforms.ToPILImage(),
                                                                                                     transforms.RandomCrop(32),
                                                                                                     transforms.RandomHorizontalFlip(),
                                                                                                     transforms.ToTensor(),
                                                                                                     transforms.Normalize(mean=cinic_mean,std=cinic_std),
                                                                                                     ]))
        train_dl = torch.utils.data.DataLoader(training_set, batch_size=train_bs, shuffle=True)
        logger.info("Len of training set: {}, len of imgs in training set: {}, len of train dl: {}".format(len(training_set), len(training_set.imgs), len(train_dl)))

        test_dl = torch.utils.data.DataLoader(torchvision.datasets.ImageFolder(cinic_directory + '/cinic-10-trainlarge/test',
            transform=transforms.Compose([transforms.ToTensor(),
            transforms.Normalize(mean=cinic_mean,std=cinic_std)])), batch_size=test_bs, shuffle=False)

    return train_dl, test_dl


def get_dataloader_dist_skew(dataset, datadir, train_bs, test_bs, dataidxs=None, gray_scale_indices_train=None, gray_scale_indices_test=None):

    if dataset in ('mnist', 'cifar10'):
        if dataset == 'mnist':
            pass
        elif dataset == 'cifar10':

            normalize_colored = transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
                                    std=[x/255.0 for x in [63.0, 62.1, 66.7]])

            normalize_gray_scale = transforms.Normalize(mean=[x/255.0 for x in [125.3, 125.3, 125.3]],
                                    std=[x/255.0 for x in [63.0, 63.0, 63.0]])


            transform_train_color = transforms.Compose([
                transforms.ToTensor(),
                transforms.Lambda(lambda x: F.pad(
                                    Variable(x.unsqueeze(0), requires_grad=False),
                                    (4,4,4,4),mode='reflect').data.squeeze()),
                transforms.ToPILImage(),
                transforms.RandomCrop(32),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize_colored,
                ])

            transform_train_gray_scale = transforms.Compose([
                transforms.ToTensor(),
                transforms.Lambda(lambda x: F.pad(
                                    Variable(x.unsqueeze(0), requires_grad=False),
                                    (4,4,4,4),mode='reflect').data.squeeze()),
                transforms.ToPILImage(),
                transforms.RandomCrop(32),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize_gray_scale,
                ])
            # data prep for test set
            transform_test_color = transforms.Compose([
                transforms.ToTensor(),
                normalize_colored])

            transform_test_gray_scale = transforms.Compose([
                transforms.ToTensor(),
                normalize_gray_scale])


            train_ds = CIFAR10ColorGrayScaleTruncated(datadir, dataidxs=dataidxs, gray_scale_indices=gray_scale_indices_train, 
                                                               train=True, transform_color=transform_train_color, 
                                                               transofrm_gray_scale=transform_train_gray_scale,
                                                               download=True)
            test_ds = CIFAR10ColorGrayScaleTruncated(datadir, gray_scale_indices=gray_scale_indices_test, 
                                                              train=False, transform_color=transform_test_color, 
                                                              transofrm_gray_scale=transform_test_gray_scale,
                                                              download=True)

            train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, shuffle=True)
            test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False)

    elif dataset == 'cinic10':
        pass
    return train_dl, test_dl


def pdm_prepare_full_weights_cnn(nets, device="cpu"):
    """
    we extract all weights of the conv nets out here:
    """
    weights = []
    for net_i, net in enumerate(nets):
        net_weights = []
        statedict = net.state_dict()

        for param_id, (k, v) in enumerate(statedict.items()):
            if device == "cpu":
                if 'fc' in k or 'classifier' in k:
                    if 'weight' in k:
                        net_weights.append(v.numpy().T)
                    else:
                        net_weights.append(v.numpy())
                elif 'conv' in k or 'features' in k:
                    if 'weight' in k:
                        _weight_shape = v.size()
                        if len(_weight_shape) == 4:
                            net_weights.append(v.numpy().reshape(_weight_shape[0], _weight_shape[1]*_weight_shape[2]*_weight_shape[3]))
                        else:
                            pass
                    else:
                        net_weights.append(v.numpy())
            else:
                if 'fc' in k or 'classifier' in k:
                    if 'weight' in k:
                        net_weights.append(v.cpu().numpy().T)
                    else:
                        net_weights.append(v.cpu().numpy())
                elif 'conv' in k or 'features' in k:
                    if 'weight' in k:
                        _weight_shape = v.size()
                        if len(_weight_shape) == 4:
                            net_weights.append(v.cpu().numpy().reshape(_weight_shape[0], _weight_shape[1]*_weight_shape[2]*_weight_shape[3]))
                        else:
                            pass
                    else:
                        net_weights.append(v.cpu().numpy())
        weights.append(net_weights)
    return weights


def pdm_prepare_weights_vggs(nets, device="cpu"):
    """
    Note that we only handle the FC parts and leave the conv layers as is
    """
    weights = []

    for net_i, net in enumerate(nets):
        layer_i = 0
        statedict = net.state_dict()

        net_weights = []
        for i, (k,v) in enumerate(statedict.items()):
            if "classifier" in k:
                if "weight" in k:
                    if device == "cpu":
                        net_weights.append(statedict[k].numpy().T)
                    else:
                        net_weights.append(statedict[k].cpu().numpy().T)
                elif "bias" in k:
                    if device == "cpu":
                        net_weights.append(statedict[k].numpy())
                    else:
                        net_weights.append(statedict[k].cpu().numpy())
        net_weights.insert(0, np.zeros(net_weights[0].shape[0], dtype=np.float32))
        weights.append(net_weights)
    return weights


def pdm_prepare_freq(cls_freqs, n_classes):
    freqs = []

    for net_i in sorted(cls_freqs.keys()):
        net_freqs = [0] * n_classes

        for cls_i in cls_freqs[net_i]:
            net_freqs[cls_i] = cls_freqs[net_i][cls_i]

        freqs.append(np.array(net_freqs))

    return freqs


def compute_ensemble_accuracy(models: list, dataloader, n_classes, train_cls_counts=None, uniform_weights=False, sanity_weights=False, device="cpu"):

    correct, total = 0, 0
    true_labels_list, pred_labels_list = np.array([]), np.array([])

    was_training = [False]*len(models)
    for i, model in enumerate(models):
        if model.training:
            was_training[i] = True
            model.eval()

    if uniform_weights is True:
        weights_list = prepare_uniform_weights(n_classes, len(models))
    elif sanity_weights is True:
        weights_list = prepare_sanity_weights(n_classes, len(models))
    else:
        weights_list = prepare_weight_matrix(n_classes, train_cls_counts)

    weights_norm = normalize_weights(weights_list)

    with torch.no_grad():
        for batch_idx, (x, target) in enumerate(dataloader):
            x, target = x.to(device), target.to(device)
            target = target.long()
            out = get_weighted_average_pred(models, weights_norm, x, device=device)

            _, pred_label = torch.max(out, 1)

            total += x.data.size()[0]
            correct += (pred_label == target.data).sum().item()

            if device == "cpu":
                pred_labels_list = np.append(pred_labels_list, pred_label.numpy())
                true_labels_list = np.append(true_labels_list, target.data.numpy())
            else:
                pred_labels_list = np.append(pred_labels_list, pred_label.cpu().numpy())
                true_labels_list = np.append(true_labels_list, target.data.cpu().numpy())

    #logger.info(correct, total)

    conf_matrix = confusion_matrix(true_labels_list, pred_labels_list)

    for i, model in enumerate(models):
        if was_training[i]:
            model.train()

    return correct / float(total), conf_matrix


class ModerateCNNContainerConvBlocks(nn.Module):
    def __init__(self, num_filters, output_dim=10):
        super(ModerateCNNContainerConvBlocks, self).__init__()

        self.conv_layer = nn.Sequential(
            # Conv Layer block 1
            nn.Conv2d(in_channels=3, out_channels=num_filters[0], kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=num_filters[0], out_channels=num_filters[1], kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Conv Layer block 2
            nn.Conv2d(in_channels=num_filters[1], out_channels=num_filters[2], kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=num_filters[2], out_channels=num_filters[3], kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.05),

            # Conv Layer block 3
            nn.Conv2d(in_channels=num_filters[3], out_channels=num_filters[4], kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=num_filters[4], out_channels=num_filters[5], kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

    def forward(self, x):
        x = self.conv_layer(x)
        return x