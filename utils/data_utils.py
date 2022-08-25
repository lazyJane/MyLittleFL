import time
import os
from utils.datasets import *

from utils.constants import *

from torch.utils.data import DataLoader

from tqdm import tqdm
from data.Mnist.split_utils import *

N_public_perclass = 1000

def get_data_dir(args):
    """
    returns a string representing the path where to find the datafile corresponding to the experiment

    :param experiment_name: name of the experiment
    :return: str

    """
    u = 0
    if 'femnist'  in args.dataset.lower():
        datatype = 'FEMnist'
        
        return os.path.join('data', datatype, "all_data")

    if 'shakespeare' in args.dataset:
        datatype = 'shakespeare'
        return os.path.join('data', datatype, "all_data")

    if 'emnist' in args.dataset.lower():
        datatype = 'EMnist'
    elif 'mnist' in args.dataset.lower():
        datatype = 'Mnist'
    elif 'cifar100' in args.dataset.lower():
        datatype = 'Cifar100'
    elif 'cifar10' in args.dataset.lower():
        datatype = 'Cifar10'
    if 'alpha' in args.dataset:
        dataset_=args.dataset.replace('alpha', '').replace('ratio', '').replace('u','').split('-')
        alpha, ratio, u =dataset_[1], dataset_[2], dataset_[3]
        data_dir = os.path.join('data', datatype, f'u{u}-alpha{alpha}-ratio{ratio}')
    else: # 'iid'  'pathological'
        dataset_=args.dataset.replace('ratio', '').replace('u','').split('-')
        split, ratio, u =dataset_[1], dataset_[2], dataset_[3]
        data_dir = os.path.join('data', datatype, f'u{u}-{split}-ratio{ratio}')
    
    #除了FEMNIST和Shakespeare都会返回u
    if int(u)>0: return data_dir, u
    else:return data_dir
    
    

def get_loaders(args, split, u, type_, root_path, data_dir, batch_size, is_validation):
    """
    constructs lists of `torch.utils.DataLoader` object from the given files in `root_path`;
     corresponding to `train_iterator`, `val_iterator` and `test_iterator`;
     `val_iterator` iterates on the same dataset as `train_iterator`, the difference is only in drop_last

    :param type_: type of the dataset;
    :param root_path: path to the data folder
    :param batch_size:
    :param is_validation: (bool) if `True` validation part is used as test
    :return:
        train_iterator, val_iterator, test_iterator
        (List[torch.utils.], List[torch.utils.DataLoader], List[torch.utils.DataLoader])

    """
    if type_ == "cifar10":
        inputs, targets = get_cifar10()
        public_batch_size=10 * N_public_perclass
    elif type_ == "cifar100":
        inputs, targets = get_cifar100()
        public_batch_size=100 * N_public_perclass
    elif type_ == "emnist":
        inputs, targets = get_emnist()
        public_batch_size=47 * N_public_perclass
    elif type_ == "mnist":
        inputs, targets = get_mnist()
        public_batch_size=10 * N_public_perclass
    else:
        inputs, targets = None, None
        if type_ == "femnist":
            public_batch_size=301
            #public_batch_size=271
        if type_ == "shakespeare":
            public_batch_size=32

    train_iterators, val_iterators, test_iterators = [], [], []
    len_tests, len_trains, len_vals = [], [], []

    public_iterator, len_public = get_loader(
                    args, 
                    split,
                    u,
                    task_id=-1,
                    type_=type_,
                    path=os.path.join(data_dir, f"public{EXTENSIONS[type_]}"),
                    batch_size=public_batch_size,
                    inputs=inputs,
                    targets=targets,
                    train=False
                )
    
    path_list=os.listdir(root_path)
    
    path_list.sort(key=lambda x:int(x[5:]))
   

    for task_id, task_dir in enumerate(tqdm(path_list)):
        task_data_path = os.path.join(root_path, task_dir)

        a = get_loader(
                args,
                split,
                u,
                task_id=task_id,
                type_=type_,
                path=os.path.join(task_data_path, f"train{EXTENSIONS[type_]}"),
                batch_size=batch_size,
                inputs=inputs,
                targets=targets,
                train=False
            ) 
        b  = get_loader(
                args,
                split,
                u,
                task_id=task_id,
                type_=type_,
                path=os.path.join(task_data_path, f"test{EXTENSIONS[type_]}"),
                batch_size=batch_size,
                inputs=inputs,
                targets=targets,
                train=False
            ) 
            
        if not a or not b:
            train_iterators.append(None)
            val_iterators.append(None)
            test_iterators.append(None)
            len_trains.append(None)
            len_tests.append(None)
            len_vals.append(None)
            continue
        train_iterator, len_train= \
            get_loader(
                args,
                split,
                u,
                task_id=task_id,
                type_=type_,
                path=os.path.join(task_data_path, f"train{EXTENSIONS[type_]}"),
                batch_size=batch_size,
                inputs=inputs,
                targets=targets,
                train=True
            )
         
        val_iterator, len_val = \
            get_loader(
                args,
                split,
                u,
                task_id=task_id,
                type_=type_,
                path=os.path.join(task_data_path, f"train{EXTENSIONS[type_]}"),
                batch_size=batch_size,
                inputs=inputs,
                targets=targets,
                train=False
            )

        if is_validation:
            test_set = "val"
        else:
            test_set = "test"

        
            test_iterator, len_test = \
                get_loader(
                    args,
                    split,
                    u,
                    task_id=task_id,
                    type_=type_,
                    path=os.path.join(task_data_path, f"{test_set}{EXTENSIONS[type_]}"),
                    batch_size=batch_size,
                    inputs=inputs,
                    targets=targets,
                    train=False
                )

        
        train_iterators.append(train_iterator)
        val_iterators.append(val_iterator)
        test_iterators.append(test_iterator)
        len_trains.append(len_train)
        len_tests.append(len_test)
        len_vals.append(len_val)

    return train_iterators, val_iterators, test_iterators, public_iterator, len_trains, len_tests, len_public


def get_loader(args, split, u, task_id, type_, path, batch_size, train, inputs=None, targets=None):
    """
    constructs a torch.utils.DataLoader object from the given path

    :param type_: type of the dataset; possible are `tabular`, `images` and `text`
    :param path: path to the data file
    :param batch_size:
    :param train: flag indicating if train loader or test loader
    :param inputs: tensor storing the input data; only used with `cifar10`, `cifar100` and `emnist`; default is None
    :param targets: tensor storing the labels; only used with `cifar10`, `cifar100` and `emnist`; default is None
    :return: torch.utils.DataLoader

    """
    
    
    if type_ == "tabular":
        dataset = TabularDataset(path)
    elif type_ == "cifar10":
        dataset = SubCIFAR10(args, split, u, task_id, path, cifar10_data=inputs, cifar10_targets=targets)
    elif type_ == "cifar100":
        dataset = SubCIFAR100(args, split, u, task_id, path, cifar100_data=inputs, cifar100_targets=targets)
    elif type_ == "mnist":
        dataset = SubMNIST(args, split, u, task_id, path, mnist_data=inputs, mnist_targets=targets)
    elif type_ == "emnist":
        dataset = SubEMNIST(args, split, u, task_id, path, emnist_data=inputs, emnist_targets=targets)
    elif type_ == "femnist":
        dataset = SubFEMNIST(path)
    elif type_ == "shakespeare":
        dataset = CharacterDataset(path, chunk_len=SHAKESPEARE_CONFIG["chunk_len"])
        #print(len(dataset))
    else:
        raise NotImplementedError(f"{type_} not recognized type; possible are {list(LOADER_TYPE.keys())}")

    if len(dataset) == 0:
        return
    
    #if task_id == 0:print(len(dataset))
    # drop last batch, because of BatchNorm layer used in mobilenet_v2
    drop_last = ((type_ == "cifar100") or (type_ == "cifar10")) and (len(dataset) > batch_size) and train

    return list(DataLoader(dataset, batch_size=batch_size, shuffle=train, drop_last=drop_last)), len(dataset)
    return DataLoader(dataset, batch_size=batch_size, shuffle=train, drop_last=drop_last, num_workers = 2)
    #有batch_normalization的话必须设置drop_last为True
    #drop_last为True 丢掉不足一个batch的数据
    #drop_last为False

