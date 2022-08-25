#!/usr/bin/env python
import argparse
from FLAlgorithms.servers.serveravg import FedAvg
from FLAlgorithms.servers.serverlocal import Fedlocal
from FLAlgorithms.servers.serverFedProx import FedProx
from FLAlgorithms.servers.serverFedCluster import FedCluster
from FLAlgorithms.servers.serverIFCA import FedIFCA
from FLAlgorithms.servers.serverFedKLtopk import FedKLtopk
from FLAlgorithms.servers.serverFedKL import FedKL
from FLAlgorithms.servers.serverFedKLEM import FedKLEM
from FLAlgorithms.servers.serverFedSEM import FedSEM
from FLAlgorithms.servers.serverperavg import PerAvg
from FLAlgorithms.servers.serverpFedMe import pFedMe
from FLAlgorithms.servers.serverFedJSEM import FedJSEM
from FLAlgorithms.servers.serverFedfuzzyC import FedfuzzyC
from FLAlgorithms.servers.serverFedKLEM_PerFedavg import FedKLEM_PerFedAvg
from FLAlgorithms.servers.serverFedKLEM_intra_cluster import FedKLEM_intra_cluster
from FLAlgorithms.servers.serverFedKLEM_intra_cluster_PerFedavg import FedKLEM_intra_cluster_PerFedavg
from FLAlgorithms.servers.serverFedKLEM_PerFedavg_ratio import FedKLEM_PerFedAvg_ratio
from FLAlgorithms.servers.serverFedKLEM_PerFedavg_ratio_e import FedKLEM_PerFedAvg_ratio_e
from utils.model_utils import *
from utils.plot_utils import *
from utils.data_utils import *
from utils.constants import *
import torch
from multiprocessing import Pool


def init_data(args, mode):
    #除了FEMNIST和Shakespeare都会返回类似 ('data/Cifar10/u80-alpha0.5-ratio1.0', '80')
    data_dir_u = get_data_dir(args)
    if len(data_dir_u) == 2: data_dir = data_dir_u[0] 
    else: data_dir = data_dir_u
    root_path=os.path.join(data_dir , mode)
    print(root_path)
    #root_path=os.path.join(data_dir , "participate")
    #root_path=os.path.join(data_dir , "unseen")
    u = 0 #用在label_swapped的时候，是participate的user数目
    if 'label_swapped' in args.dataset.lower():
        split = 'label_swapped'
        u =  int(data_dir_u[1])
    else: 
        split = '' #null表示其他

    if 'femnist' in args.dataset.lower(): #Cifar100-alpha0.5-ratio1.0-u100
        experiment = 'femnist'
    elif 'emnist' in args.dataset.lower():
        experiment = 'emnist'
        print('yea')
    elif 'mnist' in args.dataset.lower():
        experiment = 'mnist'
        print('wuwuu')
    
    elif 'cifar100' in args.dataset.lower():
        print("cifar100")
        experiment = 'cifar100'
    elif 'cifar10' in args.dataset.lower():
        print("cifar10")
        experiment = 'cifar10' 
    elif 'shakespeare' in args.dataset.lower():
        experiment = 'shakespeare'
    print("===> Building data iterators..")
    train_iterators, val_iterators, test_iterators , public_iterator, len_trains, len_tests, len_public=\
        get_loaders(
            args,
            split, 
            u,
            type_=LOADER_TYPE[experiment], #数据类型
            root_path=root_path, #  参与训练的数据目录
            data_dir=data_dir, # 总数据目录
            batch_size=args.batch_size,
            is_validation=False
        )
    return train_iterators, val_iterators, test_iterators, public_iterator, len_trains, len_tests, len_public


def create_server_n_user(args, i):
    args.device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else "cpu")
    #args.device = server_device = "cpu"
    data_participate = init_data(args, mode = "participate") 
    data_unseen = init_data(args, mode = "unseen")
    model = create_model_new(args)
    
    if ('FedKLEM_PerFedAvg_ratio_e' in args.algorithm):
        server = FedKLEM_PerFedAvg_ratio_e(args, model, data_participate, data_unseen, i) 
    elif ('FedKLEM_PerFedAvg_ratio' in args.algorithm):
        server = FedKLEM_PerFedAvg_ratio(args, model, data_participate, data_unseen, i) 
    elif('FedKLEM_intra_cluster_PerFedavg' in args.algorithm):
        server = FedKLEM_intra_cluster_PerFedavg(args, model, data_participate, data_unseen, i) 
    elif('FedKLEM_IntraC' in args.algorithm):
        server = FedKLEM_intra_cluster(args, model, data_participate, data_unseen, i) 
    elif('FedKLEM_PerFedAvg' in args.algorithm):
        server = FedKLEM_PerFedAvg(args, model, data_participate, data_unseen, i)  
    elif ('FedAvg' in args.algorithm):
        server=FedAvg(args, model, data_participate, data_unseen, i)
    elif ('FedProx' in args.algorithm):
        server = FedProx(args, model, data_participate, data_unseen, i)
    elif ('FedKLEM' in args.algorithm):
        server = FedKLEM(args, model, data_participate, data_unseen, i)    
    elif ('FedCluster' in args.algorithm):
        server = FedCluster(args, model, data_participate, data_unseen, i)
    elif ('FedIFCA' in args.algorithm):
        server = FedIFCA(args, model, data_participate, data_unseen, i)
    elif ('FedKLtopk' in args.algorithm):
        server = FedKLtopk(args, model, data_participate, data_unseen, i)
    elif ('FedKL' in args.algorithm):
        server = FedKL(args, model, data_participate, data_unseen, i)
    elif('Fedlocal' in args.algorithm):
        server = Fedlocal(args, model, data_participate, data_unseen, i)
    elif('FedSEM' in args.algorithm):
        server = FedSEM(args, model, data_participate, data_unseen, i)
    
    elif('PerFedavg' in args.algorithm):
        server = PerAvg(args, model, data_participate, data_unseen, i)
    elif('pFedME' in args.algorithm):
        server = pFedMe(args, model, data_participate, data_unseen, i)
    elif('FedJSEM' in args.algorithm):
        server = FedJSEM(args, model, data_participate, data_unseen, i)    
    elif('FedfuzzyC' in args.algorithm):
        server = FedfuzzyC(args, model, data_participate, data_unseen, i)    
      

    else:
        print("Algorithm {} has not been implemented.".format(args.algorithm))
        exit()
    return server
    
def run_job(args, i):
    torch.manual_seed(i)
    print("\n\n         [ Start training iteration {} ]           \n\n".format(i))
    # Generate model
    server = create_server_n_user(args, i)
    if args.train:
        server.train(args)
        #server.test()

def main(args):
    for i in range(args.times):
        run_job(args, i)
    print("Finished training.")


if __name__ == "__main__":
    #torch.backends.cudnn.benchmark = False
    #torch.backends.cudnn.enabled = False
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="Mnist")
    parser.add_argument("--model", type=str, default="cnn")
    parser.add_argument("--train", type=int, default=1, choices=[0,1])
    parser.add_argument("--algorithm", type=str, default="pFedMe")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--gen_batch_size", type=int, default=32, help='number of samples from generator')
    parser.add_argument("--learning_rate", type=float, default=0.01, help="Local learning rate")
    parser.add_argument("--personal_learning_rate", type=float, default=0.01, help="Personalized learning rate to caculate theta aproximately using K steps")
    parser.add_argument("--ensemble_lr", type=float, default=1e-4, help="Ensemble learning rate.")
    parser.add_argument("--beta", type=float, default=1.0, help="Average moving parameter for pFedMe, or Second learning rate of Per-FedAvg")
    parser.add_argument("--lamda", type=int, default=1, help="Regularization term")
    parser.add_argument("--mix_lambda", type=float, default=0.1, help="Mix lambda for FedMXI baseline")
    parser.add_argument("--embedding", type=int, default=0, help="Use embedding layer in generator network")
    parser.add_argument("--num_glob_iters", type=int, default=200)
    parser.add_argument("--local_epochs", type=int, default=0)
    parser.add_argument("--num_users", type=int, default=20, help="Number of Users per round")
    parser.add_argument("--K", type=int, default=1, help="Computation steps")
    parser.add_argument("--times", type=int, default=3, help="running time")
    parser.add_argument("--p", type=int, default=3, help="CLuster numbers")
    parser.add_argument("--load_model", type=bool, default=False, help="Load model or train from start")
    parser.add_argument("--result_path", type=str, default="results", help="directory path to save results")
    parser.add_argument("--E", type=int, default="0", help="fit_epochs or fit_batchs")
    parser.add_argument("--gpu", type=int, default=0, help="Which GPU to run the experiments, -1 mean CPU, 0,1,2 for GPU")
    parser.add_argument("--test_unseen", type=bool, default=False)

    args = parser.parse_args()

    print("=" * 80)
    print("Summary of training process:")
    print("Algorithm: {}".format(args.algorithm))
    print("Batch size: {}".format(args.batch_size))
    print("Learing rate       : {}".format(args.learning_rate))
    print("Ensemble learing rate       : {}".format(args.ensemble_lr))
    print("Average Moving       : {}".format(args.beta))
    print("Subset of users      : {}".format(args.num_users))
    print("Number of global rounds       : {}".format(args.num_glob_iters))
    print("Number of local rounds       : {}".format(args.local_epochs))
    print("Dataset       : {}".format(args.dataset))
    print("Local Model       : {}".format(args.model))
    print("=" * 80)
    main(args)
