from __future__ import print_function
import torch
import torchvision.transforms as transforms
import argparse
import collections
import os
import random
import sys
import pprint
import datetime
import dateutil.tz
import time
import pickle
import numpy as np
import pandas as pd
import torch.backends.cudnn as cudnn

dir_path = (os.path.abspath(os.path.join(os.path.realpath(__file__), './.')))

sys.path.append(dir_path)

from miscc.config import cfg, cfg_from_file
from torch.utils.data.sampler import SubsetRandomSampler, RandomSampler
from operator import __or__
from functools import reduce


def parse_args():
    parser = argparse.ArgumentParser(description='Train a GAN network')
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default='cfg/eval.yml', type=str)
    parser.add_argument('--gpu', dest='gpu_id', type=str, default='0')
    parser.add_argument('--data_dir', dest='data_dir', type=str, default='')
    parser.add_argument('--manualSeed', type=int, help='manual seed')
    args = parser.parse_args()
    return args

def get_label_unlabel(arr, labels_list):
    rate = 0.5
    criteria = (len(arr) * rate) // cfg.CLASSES
    print(criteria)
    # criteria = 10
    print (criteria)
    input_dict = collections.defaultdict(int)
    label = []
    for step, ylabel in enumerate(arr):
        if input_dict[int(labels_list[ylabel])] != criteria:
            input_dict[int(labels_list[ylabel])] += 1
            label.append(ylabel)
    labels_numpy = np.array(label)
    return labels_numpy, arr

def get_sampler(n_labels, labels, n=0.5):
    # Only choose digits in n_labels
    # n = number of labels per class for training
    # n_val = number of lables per class for validation
    labels_list = []
    for k, v in labels.items():
        labels_list.append(v)
    labels_numpy = np.array(labels_list)
    (indices,) = np.where(reduce(__or__, [labels_numpy == i for i in np.arange(n_labels)]))
    # ---- for CUB dataset train_test setting
    train_test_split_pth = 'put_data_here'
    df_filenames = pd.read_csv(train_test_split_pth, delim_whitespace=True, header=None)
    filenames = df_filenames[1].tolist()
    indices_train = []
    indices_test  = []
    for i in range(len(filenames)):
        # for training in CUB
        if filenames[i] == 1:
            indices_train.append(indices[i])
        # for testing in CUB
        else:
            indices_test.append(indices[i])
    indices_train = np.array(indices_train)
    indices_test  = np.array(indices_test)
    # print(indices_train)
    # ---- for CUB dataset train_test setting

    # for general setting
    # Ensure uniform distribution of labels
    # np.random.shuffle(indices)
    # indices_train = np.hstack([list(filter(lambda idx: labels_numpy[idx] == i, indices)) for i in range(n_labels)])[n:]
    # indices_test = np.hstack(
    #     [list(filter(lambda idx: labels_numpy[idx] == i, indices)) for i in range(n_labels)])[:n]

    # for semi supervised learning
    # print (indices_train.shape)

    label, unlabel  = get_label_unlabel(indices_train, labels_list)
    print (label.shape, unlabel.shape)
    indices_label   = torch.from_numpy(label)
    indices_unlabel = torch.from_numpy(unlabel)

    sampler_label   = SubsetRandomSampler(indices_label)
    sampler_unlabel = SubsetRandomSampler(indices_unlabel)
    sampler_test    = SubsetRandomSampler(indices_test)
    return sampler_label, sampler_unlabel, sampler_test

    # indices_train = torch.from_numpy(indices_train)
    # indices_test = torch.from_numpy(indices_test)
    # sampler_train = SubsetRandomSampler(indices_train)
    # sampler_test = SubsetRandomSampler(indices_test)
    # return sampler_train, sampler_test

def get_dataset(dataset, n_labels, batch_size):
    # for CUB training and testing
    label_sampler, unlabel_sampler, test_sampler = get_sampler(n_labels, dataset.labels_normal, dataset)
    label = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=label_sampler, num_workers=0,
                                           pin_memory=True, drop_last=True)
    unlabel = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=unlabel_sampler, num_workers=0,
                                           pin_memory=True, drop_last=True)
    test = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=test_sampler, num_workers=0,
                                             pin_memory=True, drop_last=True)
    return label, unlabel, test

if __name__ == "__main__":
    args = parse_args()
    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)

    if args.gpu_id != '-1':
        cfg.GPU_ID = args.gpu_id
    else:
        cfg.CUDA = False
    if args.data_dir != '':
        cfg.DATA_DIR = args.data_dir
    if cfg.TRAIN.FLAG:
        print('Using config:')
        pprint.pprint(cfg)

    if not cfg.TRAIN.FLAG:
        args.manualSeed = 918  # Change this to have different random seed during evaluation

    elif args.manualSeed is None:
        args.manualSeed = 918

    print('manualSeed is：')
    print(args.manualSeed)
    random.seed(args.manualSeed)
    np.random.seed(args.manualSeed)
    torch.manual_seed(args.manualSeed)
    torch.cuda.manual_seed(args.manualSeed)
    torch.cuda.manual_seed_all(args.manualSeed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Evaluation part
    if not cfg.TRAIN.FLAG:
        from trainers.trainer_sscgan import SSCGAN_test as evaluator
        # add coding
        imsize = cfg.TREE.BASE_SIZE * (2 ** (cfg.TREE.BRANCH_NUM-1))
        image_transform = transforms.Compose([
            transforms.Resize(int(imsize * 76 / 64)),
            transforms.RandomCrop(imsize),
            transforms.RandomHorizontalFlip(),
            ])

        from datasets import Dataset
        dataset = Dataset(cfg.DATA_DIR,
                              base_size=cfg.TREE.BASE_SIZE,
                              transform=image_transform)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=True, num_workers=0,
                                           pin_memory=True, drop_last=True)
        assert dataset
        num_gpu = len(cfg.GPU_ID.split(','))
        label, unlabel, test_d = get_dataset(dataset, cfg.CLASSES, cfg.TRAIN.BATCH_SIZE)
        algo = evaluator(unlabel, test_d)
        algo.evaluate_iccv_generate_definite_img()


    # Training part
    else:
        now = datetime.datetime.now(dateutil.tz.tzlocal())
        timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
        # ckpt file save dir
        output_dir = '/raid/SSC-GAN/output/%s_%s' % \
            (cfg.DATASET_NAME, timestamp)
        pkl_filename = 'cfg.pickle'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        with open(os.path.join(output_dir, pkl_filename), 'wb') as pk:
            pickle.dump(cfg, pk, protocol=pickle.HIGHEST_PROTOCOL)

        # Get data loader
        imsize = cfg.TREE.BASE_SIZE * (2 ** (cfg.TREE.BRANCH_NUM-1))
        image_transform = transforms.Compose([
            transforms.Resize(int(imsize * 76 / 64)),
            transforms.RandomCrop(imsize),
            transforms.RandomHorizontalFlip()])

        from datasets import Dataset
        dataset = Dataset(cfg.DATA_DIR,
                              base_size=cfg.TREE.BASE_SIZE,
                              transform=image_transform,)

        label_data, unlabel_data, test_data = get_dataset(dataset, cfg.CLASSES, cfg.TRAIN.BATCH_SIZE)


        from trainers.trainer_sscgan import SSCGAN_train as trainer
        # For semi-train
        algo = trainer(output_dir, label_data, unlabel_data, test_data, imsize)
        start_t = time.time()

        # For train
        algo.train()
        end_t = time.time()
        print('Total time for training:', end_t - start_t)