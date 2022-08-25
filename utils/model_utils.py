import json
import numpy as np
import os
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from tqdm import trange
import random
import numpy as np
from FLAlgorithms.trainmodel.models_jiu import Net, SimpleLinear
from FLAlgorithms.trainmodel.models import *
from torch.utils.data import DataLoader
from FLAlgorithms.trainmodel.generator import Generator
from utils.model_config import *
from utils.constants import *
'''
METRICS = [
    'train_acc',
    'train_loss',
    'glob_acc', #test 
    'per_acc', 
    'glob_loss', 
    'per_loss'
    ]
'''

METRICS = [
    'train_acc',
    'train_loss',
    'glob_acc', #test
    'glob_acc_bottom_1', 
    'glob_acc_bottom_2', 
    'glob_acc_bottom_3', 
    'glob_acc_bottom_4', 
    'glob_acc_bottom_5', 
    'per_acc', 
    'glob_loss', 
    'per_loss', 
    'user_train_time',
    'server_agg_time',
    'total_train_time',
    'unseen_train_acc',
    'unseen_train_loss',
    'unseen_glob_acc',
    'unseen_glob_loss',
    'unseen_glob_acc_bottom']

#METRICS = ['glob_acc', 'per_acc', 'glob_loss', 'per_loss','user_train_time', 'server_agg_time']

'''
METRICS = [
    'train_acc',
    'train_loss',
    'glob_acc', #test
    'glob_acc_bottom', 
    'per_acc', 
    'glob_loss', 
    'per_loss'
    ]
   
'''

def create_model_new(args):
    if 'femnist' in args.dataset.lower():
        print("lixiaoying")
        model = FemnistCNN(num_classes=62).to(args.device), 'cnn'
    elif 'emnist' in args.dataset.lower():
        print('lll')
        model = FemnistCNN(num_classes=47).to(args.device), 'cnn'
    elif 'mnist' in args.dataset.lower():
        #model= Net('mnist', 'cnn').to(args.device), 'cnn'
        model= Mclr().to(args.device), 'mclr'
    elif 'cifar100' in args.dataset.lower():
        model = get_mobilenet(n_classes=100).to(args.device), 'cnn'
    elif 'cifar10' in args.dataset.lower():
        model = get_mobilenet(n_classes=10).to(args.device), 'cnn'
    elif 'shakespeare' in args.dataset.lower():
        model =\
            NextCharacterLSTM(
                input_size=SHAKESPEARE_CONFIG["input_size"],
                embed_size=SHAKESPEARE_CONFIG["embed_size"],
                hidden_size=SHAKESPEARE_CONFIG["hidden_size"],
                output_size=SHAKESPEARE_CONFIG["output_size"],
                n_layers=SHAKESPEARE_CONFIG["n_layers"]
            ).to(args.device), 'lstm'
   
    return model 

def get_log_path(args, algorithm, seed, gen_batch_size=32):
    #EMnist-alpha1.0-ratio0.5_FedKLtopk_0.01_10u_32b_20_0
    alg=args.dataset + "_" + algorithm
    alg+="_" + str(args.learning_rate) + "_" + str(args.num_users)
    if args.E != 0:
        alg+="u" + "_" + str(args.batch_size) + "b" + "_" + str(args.E)
    else:
        alg+="u" + "_" + str(args.batch_size) + "b" + "_" + str(args.local_epochs)
    alg=alg + "_" + str(seed)
    if 'FedGen' in algorithm: # to accompany experiments for author rebuttal
        alg += "_embed" + str(args.embedding)
        if int(gen_batch_size) != int(args.batch_size):
            alg += "_gb" + str(gen_batch_size)
    return alg


