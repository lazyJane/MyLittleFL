import os
import pickle
import string

import torch
from torchvision.datasets import CIFAR10, CIFAR100, EMNIST, MNIST
from torchvision.transforms import Compose, ToTensor, Normalize
from torch.utils.data import Dataset

import numpy as np
from PIL import Image


class TabularDataset(Dataset):
    """
    Constructs a torch.utils.Dataset object from a pickle file;
    expects pickle file stores tuples of the form (x, y) where x is vector and y is a scalar

    Attributes
    ----------
    data: iterable of tuples (x, y)

    Methods
    -------
    __init__
    __len__
    __getitem__
    """

    def __init__(self, path):
        """
        :param path: path to .pkl file
        """
        with open(path, "rb") as f:
            self.data = pickle.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x, y = self.data[idx]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.int64), idx


class SubFEMNIST(Dataset):
    """
    Constructs a subset of FEMNIST dataset corresponding to one client;
    Initialized with the path to a `.pt` file;
    `.pt` file is expected to hold a tuple of tensors (data, targets) storing the images and there corresponding labels.

    Attributes
    ----------
    transform
    data: iterable of integers
    targets

    Methods
    -------
    __init__
    __len__
    __getitem__
    """
    def __init__(self, path):
        self.transform = Compose([
            ToTensor(),
            Normalize((0.1307,), (0.3081,))
        ])

        self.data, self.targets = torch.load(path)

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])

        img = np.uint8(img.numpy() * 255)
        img = Image.fromarray(img, mode='L')

        if self.transform is not None:
            img = self.transform(img)

        return img, target, index


class SubEMNIST(Dataset):
    """
    Constructs a subset of EMNIST dataset from a pickle file;
    expects pickle file to store list of indices

    Attributes
    ----------
    indices: iterable of integers
    transform
    data
    targets

    Methods
    -------
    __init__
    __len__
    __getitem__
    """
    def __init__(self, args, split, u, task_id, path, emnist_data=None, emnist_targets=None, transform=None):
        """
        :param path: path to .pkl file; expected to store list of indices
        :param emnist_data: EMNIST dataset inputs
        :param emnist_targets: EMNIST dataset labels
        :param transform:
        """
        with open(path, "rb") as f:
            self.indices = pickle.load(f)

        if transform is None:
            self.transform =\
                Compose([
                    ToTensor(),
                    Normalize((0.1307,), (0.3081,))
                ])

        if emnist_data is None or emnist_targets is None:
            self.data, self.targets = get_emnist()
        else:
            self.data, self.targets = emnist_data, emnist_targets

        self.data = self.data[self.indices]
        self.targets = self.targets[self.indices]

        if split == 'label_swapped':
            #mapp=np.array(['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck'])
            k = int(args.p)
            n_users = u # 总客户数量
            n_per_cluster = int(n_users / k)
            if task_id >= 0 and task_id < n_per_cluster:
                for i, target in enumerate(self.targets):
                    if target.item() == 1: self.targets[i] = torch.tensor(-1)
                    if target.item() == 2: self.targets[i] = torch.tensor(1)
                for i, target in enumerate(self.targets):
                    if target.item() == -1: self.targets[i] = torch.tensor(2)
            if task_id >= n_per_cluster and task_id < 2*n_per_cluster:
                for i, target in enumerate(self.targets):
                    if target.item() == 3: self.targets[i] = torch.tensor(-1)
                    if target.item() == 4: self.targets[i] = torch.tensor(3)
                for i, target in enumerate(self.targets):
                    if target.item() == -1: self.targets[i] = torch.tensor(4)
            if task_id >= 2*n_per_cluster and task_id < 3*n_per_cluster:
                for i, target in enumerate(self.targets):
                    if target.item() == 5: self.targets[i] = torch.tensor(-1)
                    if target.item() == 6: self.targets[i] = torch.tensor(5)
                for i, target in enumerate(self.targets):
                    if target.item() == -1: self.targets[i] = torch.tensor(6)
            if task_id >= 3*n_per_cluster and task_id < 4*n_per_cluster:
                for i, target in enumerate(self.targets):
                    if target.item() == 7: self.targets[i] = torch.tensor(-1)
                    if target.item() == 8: self.targets[i] = torch.tensor(7)
                for i, target in enumerate(self.targets):
                    if target.item() == -1: self.targets[i] = torch.tensor(8)

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])

        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        return img, target, index

class SubMNIST(Dataset):
    """
    Constructs a subset of MNIST dataset from a pickle file;
    expects pickle file to store list of indices

    Attributes
    ----------
    indices: iterable of integers
    transform
    data
    targets

    Methods
    -------
    __init__
    __len__
    __getitem__
    """

    def __init__(self, args, split, u, task_id, path, mnist_data=None, mnist_targets=None, transform=None):
        
        """
        :param path: path to .pkl file; expected to store list of indices
        :param emnist_data: EMNIST dataset inputs
        :param emnist_targets: EMNIST dataset labels
        :param transform:
        """
        with open(path, "rb") as f:
            self.indices = pickle.load(f)
        
        if transform is None:
            self.transform =\
                Compose([
                    ToTensor(),
                    Normalize((0.1307,), (0.3081,))
                ])

        if mnist_data is None or mnist_targets is None:
            self.data, self.targets = get_mnist()
        else:
            self.data, self.targets = mnist_data, mnist_targets

        self.data = self.data[self.indices]
        self.targets = self.targets[self.indices]
        #if task_id == 7 : 
            #print(self.targets)
            #input()
        
        if split == 'label_swapped':
            k = int(args.p)
            n_users = u # 总客户数量
            n_per_cluster = int(n_users / k)
            if task_id >= 0 and task_id < n_per_cluster:
                for i, target in enumerate(self.targets):
                    if target.item() == 1: self.targets[i] = torch.tensor(-1)
                    if target.item() == 2: self.targets[i] = torch.tensor(1)
                for i, target in enumerate(self.targets):
                    if target.item() == -1: self.targets[i] = torch.tensor(2)
            if task_id >= n_per_cluster and task_id < 2*n_per_cluster:
                for i, target in enumerate(self.targets):
                    if target.item() == 3: self.targets[i] = torch.tensor(-1)
                    if target.item() == 4: self.targets[i] = torch.tensor(3)
                for i, target in enumerate(self.targets):
                    if target.item() == -1: self.targets[i] = torch.tensor(4)
            if task_id >= 2*n_per_cluster and task_id < 3*n_per_cluster:
                for i, target in enumerate(self.targets):
                    if target.item() == 5: self.targets[i] = torch.tensor(-1)
                    if target.item() == 6: self.targets[i] = torch.tensor(5)
                for i, target in enumerate(self.targets):
                    if target.item() == -1: self.targets[i] = torch.tensor(6)
            if task_id >= 3*n_per_cluster and task_id < 4*n_per_cluster:
                for i, target in enumerate(self.targets):
                    if target.item() == 7: self.targets[i] = torch.tensor(-1)
                    if target.item() == 8: self.targets[i] = torch.tensor(7)
                for i, target in enumerate(self.targets):
                    if target.item() == -1: self.targets[i] = torch.tensor(8)
        

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])

        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        return img, target, index

class SubCIFAR10(Dataset):
    """
    Constructs a subset of CIFAR10 dataset from a pickle file;
    expects pickle file to store list of indices

    Attributes
    ----------
    indices: iterable of integers
    transform
    data
    targets

    Methods
    -------
    __init__
    __len__
    __getitem__
    """
    def __init__(self, args, split, u, task_id, path, cifar10_data=None, cifar10_targets=None, transform=None):
        """
        :param path: path to .pkl file; expected to store list of indices
        :param cifar10_data: Cifar-10 dataset inputs stored as torch.tensor
        :param cifar10_targets: Cifar-10 dataset labels stored as torch.tensor
        :param transform:
        """
        with open(path, "rb") as f:
            self.indices = pickle.load(f)
    
        if transform is None:
            self.transform = \
                Compose([
                    ToTensor(),
                    Normalize(
                        (0.4914, 0.4822, 0.4465),
                        (0.2023, 0.1994, 0.2010)
                    )
                ])

        if cifar10_data is None or cifar10_targets is None:
            self.data, self.targets = get_cifar10()
        else:
            self.data, self.targets = cifar10_data, cifar10_targets

        self.data = self.data[self.indices]
        self.targets = self.targets[self.indices]

        if split == 'label_swapped':
            #mapp=np.array(['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck'])
            k = int(args.p)
            n_users = u # 总客户数量
            n_per_cluster = int(n_users / k)
            if task_id >= 0 and task_id < n_per_cluster:
                for i, target in enumerate(self.targets):
                    if target.item() == 1: self.targets[i] = torch.tensor(-1)
                    if target.item() == 2: self.targets[i] = torch.tensor(1)
                for i, target in enumerate(self.targets):
                    if target.item() == -1: self.targets[i] = torch.tensor(2)
            if task_id >= n_per_cluster and task_id < 2*n_per_cluster:
                for i, target in enumerate(self.targets):
                    if target.item() == 3: self.targets[i] = torch.tensor(-1)
                    if target.item() == 4: self.targets[i] = torch.tensor(3)
                for i, target in enumerate(self.targets):
                    if target.item() == -1: self.targets[i] = torch.tensor(4)
            if task_id >= 2*n_per_cluster and task_id < 3*n_per_cluster:
                for i, target in enumerate(self.targets):
                    if target.item() == 5: self.targets[i] = torch.tensor(-1)
                    if target.item() == 6: self.targets[i] = torch.tensor(5)
                for i, target in enumerate(self.targets):
                    if target.item() == -1: self.targets[i] = torch.tensor(6)
            if task_id >= 3*n_per_cluster and task_id < 4*n_per_cluster:
                for i, target in enumerate(self.targets):
                    if target.item() == 7: self.targets[i] = torch.tensor(-1)
                    if target.item() == 8: self.targets[i] = torch.tensor(7)
                for i, target in enumerate(self.targets):
                    if target.item() == -1: self.targets[i] = torch.tensor(8)
              

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        img = Image.fromarray(img.numpy())

        if self.transform is not None:
            img = self.transform(img)

        target = target

        return img, target, index


class SubCIFAR100(Dataset):
    """
    Constructs a subset of CIFAR100 dataset from a pickle file;
    expects pickle file to store list of indices

    Attributes
    ----------
    indices: iterable of integers
    transform
    data
    targets

    Methods
    -------
    __init__
    __len__
    __getitem__
    """
    def __init__(self, args, split, u, task_id, path, cifar100_data=None, cifar100_targets=None, transform=None):
        """
        :param path: path to .pkl file; expected to store list of indices:
        :param cifar100_data: CIFAR-100 dataset inputs
        :param cifar100_targets: CIFAR-100 dataset labels
        :param transform:
        """
        with open(path, "rb") as f:
            self.indices = pickle.load(f)

        if transform is None:
            self.transform = \
                Compose([
                    ToTensor(),
                    Normalize(
                        (0.4914, 0.4822, 0.4465),
                        (0.2023, 0.1994, 0.2010)
                    )
                ])

        if cifar100_data is None or cifar100_targets is None:
            self.data, self.targets = get_cifar100()

        else:
            self.data, self.targets = cifar100_data, cifar100_targets

        self.data = self.data[self.indices]
        self.targets = self.targets[self.indices]

        if split == 'label_swapped':
            k =int(args.p)
            n_users = u # 总客户数量
            n_per_cluster = int(n_users / k)
            if task_id >= 0 and task_id < n_per_cluster:
                for i, target in enumerate(self.targets):
                    if target.item() == 1: self.targets[i] = torch.tensor(-1)
                    if target.item() == 2: self.targets[i] = torch.tensor(1)
                for i, target in enumerate(self.targets):
                    if target.item() == -1: self.targets[i] = torch.tensor(2)
            if task_id >= n_per_cluster and task_id < 2*n_per_cluster:
                for i, target in enumerate(self.targets):
                    if target.item() == 3: self.targets[i] = torch.tensor(-1)
                    if target.item() == 4: self.targets[i] = torch.tensor(3)
                for i, target in enumerate(self.targets):
                    if target.item() == -1: self.targets[i] = torch.tensor(4)
            if task_id >= 2*n_per_cluster and task_id < 3*n_per_cluster:
                for i, target in enumerate(self.targets):
                    if target.item() == 5: self.targets[i] = torch.tensor(-1)
                    if target.item() == 6: self.targets[i] = torch.tensor(5)
                for i, target in enumerate(self.targets):
                    if target.item() == -1: self.targets[i] = torch.tensor(6)
            if task_id >= 3*n_per_cluster and task_id < 4*n_per_cluster:
                for i, target in enumerate(self.targets):
                    if target.item() == 7: self.targets[i] = torch.tensor(-1)
                    if target.item() == 8: self.targets[i] = torch.tensor(7)
                for i, target in enumerate(self.targets):
                    if target.item() == -1: self.targets[i] = torch.tensor(8)

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        img = Image.fromarray(img.numpy())

        if self.transform is not None:
            img = self.transform(img)

        target = target

        return img, target, index


class CharacterDataset(Dataset):
    def __init__(self, file_path, chunk_len):
        """
        Dataset for next character prediction, each sample represents an input sequence of characters 一个样本代表人物的一个输入句子,target sequence表示input的下一个sequence
         and a target sequence of characters representing to next sequence of the input
        :param file_path: path to .txt file containing the training corpus 语料库
        :param chunk_len: (int) the length of the input and target sequences
        """
        self.all_characters = string.printable
        self.vocab_size = len(self.all_characters)
        self.n_characters = len(self.all_characters)
        self.chunk_len = chunk_len

        with open(file_path, 'r') as f:
            self.text = f.read() #读原始文本

        self.tokenized_text = torch.zeros(len(self.text), dtype=torch.long) #tensor([0., 0., 0., 0., 0.,...])

        self.inputs = torch.zeros(self.__len__(), self.chunk_len, dtype=torch.long) #两维[len(text), chunk_len)]
        self.targets = torch.zeros(self.__len__(), self.chunk_len, dtype=torch.long)

        self.__build_mapping()
        self.__tokenize()
        self.__preprocess_data()

    def __tokenize(self):
        for ii, char in enumerate(self.text):
            self.tokenized_text[ii] = self.char2idx[char] #将text文本转换为数字

    def __build_mapping(self):
        self.char2idx = dict()
        for ii, char in enumerate(self.all_characters):
            self.char2idx[char] = ii #字符转换为数字

    def __preprocess_data(self): #句子分块处理
        for idx in range(self.__len__()):
            self.inputs[idx] = self.tokenized_text[idx:idx+self.chunk_len] #
            self.targets[idx] = self.tokenized_text[idx+1:idx+self.chunk_len+1] #下一个单词

    def __len__(self):
        #print(len(self.text))
        return max(0, len(self.text) - self.chunk_len) #假如有160个字符，160-80=80

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx], idx


def get_emnist():
    """
    gets full (both train and test) EMNIST dataset inputs and labels;
    the dataset should be first downloaded (see data/emnist/README.md)
    :return:
        emnist_data, emnist_targets
    """
    emnist_path = os.path.join("data", "EMnist", "data")
    assert os.path.isdir(emnist_path), "Download EMNIST dataset!!"

    emnist_train =\
        EMNIST(
            root=emnist_path,
            split="balanced",
            download=False,
            train=True
        )

    emnist_test =\
        EMNIST(
            root=emnist_path,
            split="balanced",
            download=False,
            train=False
        )

    print(set(emnist_train.targets.numpy()))
    emnist_data =\
        torch.cat([
            emnist_train.data,
            emnist_test.data
        ])

    emnist_targets =\
        torch.cat([
            emnist_train.targets,
            emnist_test.targets
        ])

    return emnist_data, emnist_targets

def get_mnist():
    """
    gets full (both train and test) MNIST dataset inputs and labels;
    the dataset should be first downloaded (see data/emnist/README.md)
    :return:
        emnist_data, emnist_targets
    """
    mnist_path = os.path.join("data", "Mnist", "data")
    print(mnist_path)
    assert os.path.isdir(mnist_path), "Download MNIST dataset!!"

    mnist_train =\
        MNIST(
            root=mnist_path,
            download=True,
            train=True
        )

    mnist_test =\
        MNIST(
            root=mnist_path,
            download=True,
            train=False
        )

    mnist_data =\
        torch.cat([
            mnist_train.data,
            mnist_test.data
        ])

    mnist_targets =\
        torch.cat([
            mnist_train.targets,
            mnist_test.targets
        ])

    return mnist_data, mnist_targets

def get_cifar10():
    """
    gets full (both train and test) CIFAR10 dataset inputs and labels;
    the dataset should be first downloaded (see data/emnist/README.md)
    :return:
        cifar10_data, cifar10_targets
    """
    cifar10_path = os.path.join("data", "Cifar10", "raw_data")
    assert os.path.isdir(cifar10_path), "Download cifar10 dataset!!"

    cifar10_train =\
        CIFAR10(
            root=cifar10_path,
            train=True, download=False
        )

    cifar10_test =\
        CIFAR10(
            root=cifar10_path,
            train=False,
            download=False)

    cifar10_data = \
        torch.cat([
            torch.tensor(cifar10_train.data),
            torch.tensor(cifar10_test.data)
        ])

    cifar10_targets = \
        torch.cat([
            torch.tensor(cifar10_train.targets),
            torch.tensor(cifar10_test.targets)
        ])

    return cifar10_data, cifar10_targets


def get_cifar100():
    """
    gets full (both train and test) CIFAR100 dataset inputs and labels;
    the dataset should be first downloaded (see data/cifar100/README.md)
    :return:
        cifar100_data, cifar100_targets
    """
    cifar100_path = os.path.join("data", "Cifar100", "raw_data")
    assert os.path.isdir(cifar100_path), "Download Cifar100 dataset!!"

    cifar100_train =\
        CIFAR100(
            root=cifar100_path,
            train=True, download=False
        )

    cifar100_test =\
        CIFAR100(
            root=cifar100_path,
            train=False,
            download=False)

    cifar100_data = \
        torch.cat([
            torch.tensor(cifar100_train.data),
            torch.tensor(cifar100_test.data)
        ])

    cifar100_targets = \
        torch.cat([
            torch.tensor(cifar100_train.targets),
            torch.tensor(cifar100_test.targets)
        ])

    return cifar100_data, cifar100_targets
