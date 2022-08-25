from FLAlgorithms.users.userFedKLtopk import UserFedKLtopk
from FLAlgorithms.servers.serverbase import *
from FLAlgorithms.trainmodel.models_jiu import Net, SimpleLinear
from utils.model_utils import *
import numpy as np
import matplotlib.pyplot as plt
# Implementation for FedAvg Server
import time
import torch
import torch.nn.functional as F
from tqdm import tqdm
import pandas as pd
import torch
class FedKLtopk(Server):
    def __init__(self, args, model, data_participate, data_unseen, seed):
        super().__init__(args, model, data_participate, data_unseen, seed)

        #self.p = 3
        #self.models = [create_model_new(args)[0] for p_i in range(self.p)]
        for task_id, (train_iterator, val_iterator, test_iterator, len_train, len_test) in \
            enumerate(tqdm(zip(self.train_iterators, self.val_iterators, self.test_iterators, self.len_trains, self.len_tests), total=len(self.train_iterators))):
            if train_iterator is None or test_iterator is None:
                continue
            #GPU_idx = total_users % 6
            #user_device = torch.device("cuda:{}".format(GPU_idx) if torch.cuda.is_available() and args.gpu != -1 else "cpu")
            user = UserFedKLtopk(args, task_id, model, train_iterator, val_iterator, test_iterator, len_train, len_test, self.len_public, use_adam=False)
            self.users.append(user)
            self.total_train_samples += user.train_samples
        print("Number of users / total users:",args.num_users, " / " , self.total_users)
        print("Finished creating FedKLtopk server.")

        self.models = [create_model_new(args)[0] for i in range(self.total_users)]

        self.cluster_indices = [np.arange(self.num_users).astype("int")] 
        self.user_clusters = [[self.users[i] for i in idcs] for idcs in self.cluster_indices]

    def flatten(self, source):
            return torch.cat([value.flatten() for value in source.values()]) #dim默认为0

    def get_logits_clients(self, selected_users):
        logits = [] 
        for i, user in enumerate(selected_users):
             with torch.no_grad():
                for x, y,_ in self.public_loader: 
                    x, y = x.to(self.device), y.to(self.device)
                    if 'cifar' in self.dataset: 
                        logits.append(user.model(x))
                    else:
                        logits.append(user.model(x)['logit']) #[tensor(),...tensor()]
        #print(len(logits))
        #input()
        return logits
        
    def compute_pairwise_KL(self, sources):
            angles = torch.zeros([len(sources), len(sources)])
            for i, source1 in enumerate(sources):
                 for j, source2 in enumerate(sources):
                    #print(source1)
                    #print(source1.shape)
                    #input()
                    angles[i,j] = F.kl_div(F.log_softmax(source1, dim=1), F.softmax(source2, dim=1))
            #print(angles.numpy())
            #input()
            return angles.numpy()

    def compute_pairwise_JS(self, sources):
            angles = torch.zeros([len(sources), len(sources)])
            for i, source1 in enumerate(sources):
                 for j, source2 in enumerate(sources):
                    p_output = F.softmax(source1, dim=1)
                    q_output = F.softmax(source2, dim=1)
                    log_mean_output = ((p_output + q_output )/2).log()
                    angles[i,j] = (F.kl_div(log_mean_output, p_output) + F.kl_div(log_mean_output, q_output)) / 2
            data = pd.DataFrame(angles.numpy())

            writer = pd.ExcelWriter('./similarity matrix.xlsx')		# 写入Excel文件
            data.to_excel(writer, 'page_1', float_format='%.5f')		# ‘page_1’是写入excel的sheet名
            writer.save()

            return angles.numpy()


    def aggregate_topk_parameters(self, similarities, glob_iter):
        top_idx = np.zeros((len(similarities),len(similarities)))
        for i, user in enumerate(self.selected_users):
            top_idx[i] = np.argsort(similarities[i]) #kl距离越小，相似度越高  #[i,j] top_idx Ci与其他客户端的相似度从高到低
            
        total_train = 0
        for user in self.selected_users:
            total_train += user.train_samples
        
        a = float(0.8/int(self.num_users / 2))
        b = float(0.2/int(self.num_users / 2))

        ratio = 0.1
        for i, user in enumerate(self.selected_users):
            for param in self.models[i].parameters():
                #param.data = user.model.data #存储客户端模型参数
                param.data = torch.zeros_like(param.data)  # 服务器端存储的客户端i的模型重置为0
            for idx, user_id in enumerate(top_idx[i]): #对客户端i有益的客户
                user = self.selected_users[user_id.astype(int)]
                if idx < int(self.num_users / 2):
                    self.add_topk_parameters(self.models[i], user.model, a, ratio) 
                else :
                    self.add_topk_parameters(self.models[i], user.model, b, ratio) 
       
        
    def add_topk_parameters(self, self_model, other_model, a, ratio):
        for self_param, other_param in zip(self_model.parameters(), other_model.parameters()):
            self_param.data = self_param.data + other_param.data.clone() * a 

    def send_topk_parameters(self):
        for i, user in enumerate(self.selected_users):
            user.set_parameters(self.models[i])


    def train(self, args):
        for glob_iter in range(self.num_glob_iters):
            print("-------------Round number: ",glob_iter, " -------------")
            self.selected_users = self.select_users(glob_iter,self.num_users)
            self.send_parameters(self.users, mode=self.mode) #客户端接受此轮起始参数点

            self.timestamp = time.time() # log user-training + compute dW + similarities start time train+计算dW时间
            for user in self.selected_users: # allow selected users to train
                user.compute_weight_update(glob_iter, self.personalized)#loss

            logits = self.get_logits_clients(self.selected_users)
            similarities = self.compute_pairwise_JS(logits)##shape[10,10] 计算相似度
            
            #更新客户端(个性化)模型 更新模型
            self.aggregate_topk_parameters(similarities, glob_iter) 
            self.send_topk_parameters() 
            self.evaluate()

            self.aggregate_parameters()  #用个性化模型更新全局模型
           
            self.save_results(args)
            #self.save_model()