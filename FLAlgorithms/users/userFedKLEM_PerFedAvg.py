import torch
from FLAlgorithms.users.userbase import User
from FLAlgorithms.optimizers.fedoptimizer import MySGD
import copy
class UserFedKLEM_PerFedAvg(User):
    def __init__(self,  args, id, model, train_iterator, val_iterator, test_iterator, len_train, len_test, len_public, use_adam=False):
        super().__init__(args, id, model, train_iterator, val_iterator, test_iterator, len_train, len_test, len_public, use_adam=False)

        self.W = {key : value for key, value in self.model.named_parameters()}
        self.dW = {key : torch.zeros_like(value) for key, value in self.model.named_parameters()}
        self.W_old = {key : torch.zeros_like(value) for key, value in self.model.named_parameters()}
        if 'mnist' == self.dataset[0]:
            self.optimizer = MySGD(self.model.parameters(), lr=self.learning_rate)
        else:
            self.optimizer = MySGD(self.model.parameters(), lr=self.learning_rate,  momentum=0.9)
        self.cluster_idx = 0
        #self.optimizer = torch.optim.SGD(params=self.model.parameters(), lr=0.1, momentum=0.9)
        #self.public_loader , self.iter_proxyloader= read_proxy_data(public_data)

    def set_cluster_idx(self, cluster_idx):
        self.cluster_idx = cluster_idx

    def train(self, glob_iter, personalized=False, lr_decay=True, count_labels=True):
       
        #local_epochs = int(self.E * (self.train_samples / self.batch_size ))
        
        if self.E == 0:
            local_epochs = self.local_epochs
        else:
            if self.train_samples < self.batch_size:
                local_epochs = 1
            else:
                local_epochs = int(self.E * (self.train_samples / self.batch_size ) )
        
        for epoch in range(1, local_epochs + 1):  # local update 
            self.model.train()

            temp_model = copy.deepcopy(list(self.model.parameters()))

                #step 1
            result = self.get_next_train_batch()
            X, y = result['X'], result['y']
            X, y = X.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()
            if 'cifar10' in self.dataset or 'cifar100' in self.dataset or 'shakespeare' in self.dataset:   
                output = self.model(X)
                loss = self.ce_loss(output, y)
            else: 
                output = self.model(X)['output']
                loss = self.loss(output, y)
            loss.backward()
            self.optimizer.step()

                #step 2
            result = self.get_next_train_batch()
            X, y = result['X'], result['y']
            X, y = X.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()
            if 'cifar10' in self.dataset or 'cifar100' in self.dataset or 'shakespeare' in self.dataset:   
                output = self.model(X)
                loss = self.ce_loss(output, y)
            else: 
                output = self.model(X)['output']
                loss = self.loss(output, y)
            loss.backward()

                # restore the model parameters to the one before first update
            for old_p, new_p in zip(self.model.parameters(), temp_model):
                old_p.data = new_p.data.clone()
                
            self.optimizer.step(beta = self.beta)

                # clone model to user model 
            self.clone_model_paramenter(self.model.parameters(), self.local_model)    


    def train_one_step(self):
        self.model.train()
        #step 1
        if 'cifar10' in self.dataset or 'cifar100' in self.dataset or 'shakespeare' in self.dataset:   
            result = self.get_next_train_batch()
        else:
            result = self.get_next_train_batch()
        X, y = result['X'], result['y']
        X, y = X.to(self.device), y.to(self.device)
        self.optimizer.zero_grad()
        if 'cifar10' in self.dataset or 'cifar100' in self.dataset or 'shakespeare' in self.dataset:   
            output = self.model(X)
            loss = self.ce_loss(output, y)
        else: 
            output = self.model(X)['output']
            loss = self.loss(output, y)
        loss.backward()
        self.optimizer.step()
            #step 2
        if 'cifar10' in self.dataset or 'cifar100' in self.dataset or 'shakespeare' in self.dataset:   
            result = self.get_next_train_batch()
        else:
            result = self.get_next_train_batch()
        X, y = result['X'], result['y']
        X, y = X.to(self.device), y.to(self.device)
        self.optimizer.zero_grad()
        if 'cifar10' in self.dataset or 'cifar100' in self.dataset or 'shakespeare' in self.dataset:   
            output = self.model(X)
            loss = self.ce_loss(output, y)
        else: 
            output = self.model(X)['output']
            loss = self.loss(output, y)
        loss.backward()
        self.optimizer.step(beta=self.beta)

    def train_unseen(self, glob_iter, personalized=False, lr_decay=True, count_labels=True):
        #self.clean_up_counts()
        #self.model = copy.deepcopy()
        if self.E != 0: 
            self.fit_epochs(glob_iter, lr_decay=True)
        else: 
            self.fit_batches(glob_iter, count_labels=True, lr_decay=True)

    