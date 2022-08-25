import torch
from FLAlgorithms.users.userbase import User

class UserFedKL(User):
    def __init__(self,  args, id, model, train_iterator, val_iterator, test_iterator, use_adam=False):
        super().__init__(args, id, model, train_iterator, val_iterator, test_iterator, use_adam=use_adam)
        
        self.W = {key : value for key, value in self.model.named_parameters()}
        self.dW = {key : torch.zeros_like(value) for key, value in self.model.named_parameters()}
        self.W_old = {key : torch.zeros_like(value) for key, value in self.model.named_parameters()}
        #self.optimizer = torch.optim.SGD(params=self.model.parameters(), lr=0.1, momentum=0.9)
        #self.public_loader , self.iter_proxyloader= read_proxy_data(public_data)

    def update_label_counts(self, labels, counts):
        for label, count in zip(labels, counts):
            self.label_counts[int(label)] += count

    def clean_up_counts(self):
        del self.label_counts
        self.label_counts = {int(label):1 for label in range(self.unique_labels)}

    def copy(self, target, source):
        for name in target:
            target[name].data = source[name].data.clone()

    def subtract_(self, target, minuend, subtrahend):
        for name in target:
            target[name].data = minuend[name].data.clone()-subtrahend[name].data.clone()

    def reset(self):
        self.copy(target=self.W, source=self.W_old)

    def train(self, glob_iter, personalized=False, lr_decay=True, count_labels=True):
        self.clean_up_counts()
        #self.model = copy.deepcopy()
        if self.E != 0: 
            self.fit_epochs(glob_iter, lr_decay=True)
        else: 
            self.fit_batches(glob_iter, count_labels=True, lr_decay=True)
        


    def compute_weight_update(self, glob_iter, personalized):
        self.copy(target=self.W_old, source=self.W)
        #self.optimizer.param_groups[0]["lr"]*=0.99
        self.train(glob_iter, personalized) 
        self.subtract_(target=self.dW, minuend=self.W, subtrahend=self.W_old)  

    