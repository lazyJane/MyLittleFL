import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import json
from torch.utils.data import DataLoader
import numpy as np
import copy
from utils.model_config import RUNCONFIGS
from FLAlgorithms.optimizers.fedoptimizer import pFedIBOptimizer
from utils.constants import *
from utils.model_utils import *


class User:
    """
    Base class for users in federated learning.
    """
    def __init__(
            self, args, id, model, train_iterator, val_iterator, test_iterator, len_train, len_test, len_public, use_adam=False):
            
        self.device = args.device
        self.model = copy.deepcopy(model[0])
        self.model_name = model[1]
        self.id = id  # integer
        self.train_samples = len_train
        self.test_samples = len_test
        self.public_samples = len_public
        self.batch_size = args.batch_size
        self.learning_rate = args.learning_rate
        self.beta = args.beta
        self.lamda = args.lamda
        self.local_epochs = args.local_epochs
        self.E = args.E
        self.algorithm = args.algorithm
        self.K = args.K
        self.dataset = args.dataset.lower().replace('alpha', '').replace('ratio', '').replace('u','').split('-')#['emnist', '0.5', '0.1', '100']
        self.trainloader = train_iterator
        self.valoader =  val_iterator
        self.testloader = test_iterator
        #self.trainloader = DataLoader(train_data, self.batch_size, drop_last=False)
        #self.testloader =  DataLoader(test_data, self.batch_size, drop_last=False)
        #self.testloaderfull = DataLoader(test_data, self.test_samples)
        #self.trainloaderfull = DataLoader(train_data, self.train_samples)
        self.iter_trainloader = iter(self.trainloader)
        self.iter_testloader = iter(self.testloader)
        #dataset_name = get_dataset_name(self.dataset)
        #self.unique_labels = RUNCONFIGS[dataset_name]['unique_labels']
        #self.generative_alpha = RUNCONFIGS[dataset_name]['generative_alpha']
        #self.generative_beta = RUNCONFIGS[dataset_name]['generative_beta']

        # those parameters are for personalized federated learning.
        self.local_model = copy.deepcopy(list(self.model.parameters()))
        self.personalized_model_bar = copy.deepcopy(list(self.model.parameters()))
        self.prior_decoder = None
        self.prior_params = None

        self.init_loss_fn()
        if use_adam:
            self.optimizer=torch.optim.Adam(
                params=self.model.parameters(),
                lr=self.learning_rate, betas=(0.9, 0.999),
                eps=1e-08, weight_decay=1e-2, amsgrad=False)
        else:
            if 'mnist' == self.dataset[0]:
                self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)
            else:  
                self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate,  momentum=0.9)
        self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=self.optimizer, gamma=0.99) # 学习率指数衰减
        self.label_counts = {}

    def init_loss_fn(self):
        self.loss=nn.NLLLoss()
        self.dist_loss = nn.MSELoss()
        self.ensemble_loss=nn.KLDivLoss(reduction="batchmean")
        self.ce_loss = nn.CrossEntropyLoss()

        all_characters = string.printable
        labels_weight = torch.ones(len(all_characters), device=self.device)
        for character in CHARACTERS_WEIGHTS:
            labels_weight[all_characters.index(character)] = CHARACTERS_WEIGHTS[character] #第i个字符对应的label_weights
        labels_weight = labels_weight * 8
        self.shakespeare_loss = nn.CrossEntropyLoss(reduction="none", weight=labels_weight).to(self.device)
        

    def set_parameters(self, model,beta=1):
        for old_param, new_param, local_param in zip(self.model.parameters(), model.parameters(), self.local_model):
            if beta == 1:
                old_param.data = new_param.data.clone()
                local_param.data = new_param.data.clone()
            else:
                old_param.data = beta * new_param.data.clone() + (1 - beta)  * old_param.data.clone()
                local_param.data = beta * new_param.data.clone() + (1-beta) * local_param.data.clone()
    
    def set_uij_parameters(self, models, u, user_idx):
        for param in self.model.parameters():
            param.data = torch.zeros_like(param.data)
        for cluster_idx, cluster_model in enumerate(models):
            for user_param, cluster_param in zip(self.model.parameters(), cluster_model.parameters()):
                user_param.data += cluster_param.data.clone() * u[user_idx][cluster_idx]

    def set_prior_decoder(self, model, beta=1):
        for new_param, local_param in zip(model.personal_layers, self.prior_decoder):
            if beta == 1:
                local_param.data = new_param.data.clone()
            else:
                local_param.data = beta * new_param.data.clone() + (1 - beta) * local_param.data.clone()


    def set_prior(self, model):
        for new_param, local_param in zip(model.get_encoder() + model.get_decoder(), self.prior_params):
            local_param.data = new_param.data.clone()

    # only for pFedMAS
    def set_mask(self, mask_model):
        for new_param, local_param in zip(mask_model.get_masks(), self.mask_model.get_masks()):
            local_param.data = new_param.data.clone()

    def set_shared_parameters(self, model, mode='decode'):
        # only copy shared parameters to local
        for old_param, new_param in zip(
                self.model.get_parameters_by_keyword(mode),
                model.get_parameters_by_keyword(mode)
        ):
            old_param.data = new_param.data.clone()   

    


    def get_parameters(self):
        for param in self.model.parameters():
            param.detach()
        return self.model.parameters()


    def clone_model_paramenter(self, param, clone_param):
        with torch.no_grad():
            for param, clone_param in zip(param, clone_param):
                clone_param.data = param.data.clone()
        return clone_param
    
    def get_updated_parameters(self):
        return self.local_weight_updated
    
    def update_parameters(self, new_params, keyword='all'):
        for param , new_param in zip(self.model.parameters(), new_params):
            param.data = new_param.data.clone()

    def get_grads(self):
        grads = []
        for param in self.model.parameters():
            if param.grad is None:
                grads.append(torch.zeros_like(param.data))
            else:
                grads.append(param.grad.data)
        return grads

    def fit_epochs(self, glob_iter = 0, lr_decay = True, unseen = False):
        if 'shakespeare' in self.dataset: 
            self.fit_epochs_shakespeare(glob_iter, lr_decay)
            return 

        self.model.train()
        for step in range(self.E):
            for x, y, _ in self.trainloader:
                x, y = x.to(self.device), y.to(self.device)
                
                self.optimizer.zero_grad()
                if 'cifar10' in self.dataset or 'cifar100' in self.dataset or 'shakespeare' in self.dataset:   
                    output=self.model(x)
                    loss=self.ce_loss(output, y)
                else:
                    output=self.model(x)['output']
                    loss=self.loss(output, y)
                loss.backward()
                self.optimizer.step()#self.plot_Celeb)
            if lr_decay:
                if unseen:
                    self.lr_scheduler.step(step)
                else:
                    self.lr_scheduler.step(glob_iter) #存疑
                


    def fit_batches(self, glob_iter = 0, count_labels=False, lr_decay=True, unseen = False):
        self.model.train()
        for epoch in range(1, self.local_epochs + 1):
            for i in range(self.K):
                result =self.get_next_train_batch(count_labels=count_labels)
                X, y = result['X'], result['y']
                X, y = X.to(self.device), y.to(self.device)
                #if count_labels:
                    #self.update_label_counts(result['labels'], result['counts'])

                self.optimizer.zero_grad()
                if 'cifar10' in self.dataset or 'cifar100' in self.dataset or 'shakespeare' in self.dataset:   
                    output=self.model(X)
                    loss=self.ce_loss(output, y)
                else:
                    output=self.model(X)['output']
                    loss=self.loss(output, y)
                loss.backward()
                self.optimizer.step()#self.plot_Celeb)
        if lr_decay:
            if unseen:
                    self.lr_scheduler.step(epoch)
            else:
                    self.lr_scheduler.step(glob_iter) #存疑

    def fit_epochs_shakespeare(self, glob_iter, lr_decay):
        self.model.train()
        for step in range(self.E):
            for x, y, indices in self.trainloader:
                x = x.to(self.device)
                y = y.to(self.device)

               # n_samples += y.size(0)
                chunk_len = y.size(1)
                self.optimizer.zero_grad()

                y_pred = self.model(x)
                loss_vec = self.shakespeare_loss(y_pred, y)

                #if weights is not None:
                    #weights = weights.to(self.device)
                    #loss = (loss_vec.T @ weights[indices]).mean() / loss_vec.size(0)
                #else:
                loss = loss_vec.mean()

                loss.backward()

                self.optimizer.step()

                #global_loss += loss.detach() * loss_vec.size(0) / chunk_len
                #global_metric += self.metric(y_pred, y).detach() / chunk_len
            #return global_loss / n_samples, global_metric / n_samples

    def train_error_and_loss(self):
        if 'shakespeare' in self.dataset: 
            return self.train_error_and_loss_shakespeare()
        self.model.eval()
        train_acc = 0
        loss = 0
        train_samples = 0
        with torch.no_grad():
            for x, y, _ in self.trainloader:
                x, y = x.to(self.device), y.to(self.device)
                if 'cifar10' in self.dataset or 'cifar100' in self.dataset:     
                    output = self.model(x)
                    loss += self.ce_loss(output, y)
                else: 
                    output = self.model(x)['output']
                    loss += self.loss(output, y)
                train_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
                train_samples += y.shape[0]
                #print(self.id + ", Train Accuracy:", train_acc)
                #print(self.id + ", Train Loss:", loss)
        return train_acc, loss , train_samples

    def train_error_and_loss_shakespeare(self):
        self.model.eval()
        train_acc = 0
        loss = 0
        test_samples = 0
        with torch.no_grad():
            for x, y,_ in self.trainloader:
                x = x.to(self.device)
                y = y.to(self.device)
                test_samples += y.size(0)

                chunk_len = y.size(1)
                y_pred = self.model(x)

                loss += self.shakespeare_loss(y_pred, y).sum().detach() / chunk_len
                #test_acc += self.metric(y_pred, y).detach() / chunk_len
                train_acc += (torch.sum(torch.argmax(y_pred, dim=1) == y)).item()/ chunk_len

        return  train_acc, loss, test_samples

    def test(self):
        if 'shakespeare' in self.dataset: 
            return self.test_shakespeare()
        self.model.eval()
        test_acc = 0
        loss = 0
        test_samples = 0
        with torch.no_grad():
            for x, y,_ in self.testloader:
                x, y = x.to(self.device), y.to(self.device)
                if 'cifar10' in self.dataset or 'cifar100' in self.dataset:       
                    output = self.model(x)
                    loss += self.ce_loss(output, y)
                else: 
                    output = self.model(x)['output']
                    loss += self.loss(output, y)
                test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
                test_samples += y.shape[0]
        return test_acc, loss, test_samples

    def test_shakespeare(self):
        self.model.eval()
        test_acc = 0
        loss = 0
        test_samples = 0
        with torch.no_grad():
            for x, y,_ in self.testloader:
                x = x.to(self.device)
                y = y.to(self.device)
                test_samples += y.size(0)

                chunk_len = y.size(1)
                y_pred = self.model(x)

                loss += self.shakespeare_loss(y_pred, y).sum().detach() / chunk_len
                #test_acc += self.metric(y_pred, y).detach() / chunk_len
                test_acc += (torch.sum(torch.argmax(y_pred, dim=1) == y)).item()/ chunk_len

        return  test_acc, loss, test_samples

    def test_personalized_model(self):
        self.model.eval()
        test_acc = 0
        loss = 0
        test_samples = 0
        self.update_parameters(self.personalized_model_bar)
        with torch.no_grad():
            for x, y,_ in self.testloader:
                x, y = x.to(self.device), y.to(self.device)
                if 'cifar10' in self.dataset or 'cifar100' in self.dataset:     
                    output = self.model(x)
                    loss += self.ce_loss(output, y)
                else: 
                    output = self.model(x)['output']
                    loss += self.loss(output, y)
                test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
                #@loss += self.loss(output, y)
                #print(self.id + ", Test Accuracy:", test_acc / y.shape[0] )
                #print(self.id + ", Test Loss:", loss)
                test_samples += y.shape[0]
        self.update_parameters(self.local_model)
        return test_acc, loss, test_samples
    
    def train_error_and_loss_personalized_model(self):
        self.model.eval()
        train_acc = 0
        loss = 0
        train_samples = 0
        self.update_parameters(self.personalized_model_bar)
        with torch.no_grad():
            for x, y,_ in self.trainloader:
                x, y = x.to(self.device), y.to(self.device)
                if 'cifar10' in self.dataset or 'cifar100' in self.dataset:
                    output = self.model(x)
                    loss += self.ce_loss(output, y)
                else: 
                    output = self.model(x)['output']
                    loss += self.loss(output, y)
                train_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
                train_samples += y.shape[0]
                #print(self.id + ", Train Accuracy:", train_acc)
                #print(self.id + ", Train Loss:", loss)
        self.update_parameters(self.local_model)
        return train_acc, loss, train_samples


    def get_next_train_batch(self, count_labels=False):
        try:
            # Samples a new batch for personalizing
            (X, y, _) = next(self.iter_trainloader)
        except StopIteration:
            # restart the generator if the previous generator is exhausted.
            self.iter_trainloader = iter(self.trainloader)
            (X, y, _) = next(self.iter_trainloader)
        result = {'X': X, 'y': y}
        '''
        if count_labels:
            unique_y, counts=torch.unique(y, return_counts=True)
            unique_y = unique_y.detach().numpy()
            counts = counts.detach().numpy()
            result['labels'] = unique_y
            result['counts'] = counts
        '''
        return result

    def get_next_test_batch(self):
        try:
            # Samples a new batch for personalizing
            (X, y,_) = next(self.iter_testloader)
        except StopIteration:
            # restart the generator if the previous generator is exhausted.
            self.iter_testloader = iter(self.testloader)
            (X, y,_) = next(self.iter_testloader)
        result = {'X': X, 'y': y}
        return result

    def save_model(self):
        model_path = os.path.join("models", self.dataset)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        torch.save(self.model, os.path.join(model_path, "user_" + self.id + ".pt"))

    def load_model(self):
        model_path = os.path.join("models", self.dataset)
        self.model = torch.load(os.path.join(model_path, "server" + ".pt"))

    def load_user_model(self):
        model_path = os.path.join("models", self.dataset)
        self.model = torch.load(os.path.join(model_path, "user" + self.id + ".pt"))
    
    @staticmethod
    def model_exists():
        return os.path.exists(os.path.join("models", "server" + ".pt"))

    def train_unseen(self, glob_iter, personalized=False, lr_decay=True, count_labels=True):
        #self.clean_up_counts()
        #self.model = copy.deepcopy()
        if self.E != 0: 
            self.fit_epochs(lr_decay=True, unseen = True)
        else: 
            self.fit_batches(lr_decay=True, unseen = True)

    