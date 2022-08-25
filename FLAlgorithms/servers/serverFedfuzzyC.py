from enum import EnumMeta
from FLAlgorithms.users.userFedKLEM import UserFedKLEM
from FLAlgorithms.servers.serverbase import Server
from FLAlgorithms.trainmodel.models_jiu import Net, SimpleLinear
from utils.model_utils import *
import numpy as np
import matplotlib.pyplot as plt
# Implementation for FedAvg Server
import time
import torch
import torch.nn.functional as F
torch.manual_seed(0)
from tqdm import tqdm

class FedfuzzyC(Server):
    def __init__(self, args, model, data_participate, data_unseen, seed):
        super().__init__(args, model, data_participate, data_unseen, seed)

        self.p = 3
        self.models = [create_model_new(args)[0] for p_i in range(self.p)]
        self.c =  torch.zeros([args.num_users, self.p])

        for task_id, (train_iterator, val_iterator, test_iterator, len_train, len_test) in \
            enumerate(tqdm(zip(self.train_iterators, self.val_iterators, self.test_iterators, self.len_trains, self.len_tests), total=len(self.train_iterators))):
            if train_iterator is None or test_iterator is None:
                continue
            user = UserFedKLEM(args, task_id, model, train_iterator, val_iterator,  test_iterator, len_train, len_test, self.len_public, use_adam=False)
            self.users.append(user)
            self.total_train_samples += user.train_samples
        print("Number of users / total users:",args.num_users, " / " , self.total_users)
        print("Finished creating FedfuzzyC server.")



    def get_logits_clients(self, users): 
        logits = []
        for i, user in enumerate(users):
            with torch.no_grad():
                for x, y,_ in self.public_loader: 
                    x, y = x.to(self.device), y.to(self.device)
                    if 'cifar' in self.dataset or 'shakespeare' in self.dataset: 
                        logits.append(user.model(x))
                    else:
                        logits.append(user.model(x)['logit']) #[tensor(),...tensor()] 每个用户的logits
        return logits

    def get_logits_centers(self):
        logits = []
        for i in range(self.p):
            with torch.no_grad():
                    for x, y,_ in self.public_loader: 
                        x, y = x.to(self.device), y.to(self.device)
                        if 'cifar' in self.dataset or 'shakespeare' in self.dataset: 
                            #print(self.models[i](x).shape)
                            logits.append(self.models[i](x)) # torch.Size([32, 100, 80])
                        else:
                            #print(self.models[i](x)['logit'].shape)
                            logits.append(self.models[i](x)['logit']) #[tensor(),...tensor()]
        return logits   

    def get_u(self, logits_users, logits_centers): #[tensor(),...tensor()]
        u = np.zeros([len(logits_users), len(logits_centers)]) #隶属度矩阵
        JS_dist = torch.zeros([len(logits_users), len(logits_centers)]) #距离矩阵
        KL_dist = torch.zeros([len(logits_users), len(logits_centers)]) #距离矩阵
        for i, logits_i in enumerate(logits_users):
            for j, logits_j in enumerate(logits_centers):
                p_output = F.softmax(logits_i, dim=1)
                q_output = F.softmax(logits_j, dim=1)
                log_mean_output = ((p_output + q_output )/2).log()
                JS_dist[i,j] = (F.kl_div(log_mean_output, p_output) + F.kl_div(log_mean_output, q_output)) / 2
                #KL_dist[i,j] = F.kl_div(F.log_softmax(logits_i, dim=1), F.softmax(logits_j, dim=1), reduction="batchmean")
        print(JS_dist)
        #input()
        for i, logits_i in enumerate(logits_users):
            for j, logits_j in enumerate(logits_centers): #计算第i个用户对第j个簇的隶属度
                #计算分母
                s = 0
                for k, logits_k in enumerate(logits_centers):
                    s_jk = (JS_dist[i,j] * JS_dist[i,j]) / (JS_dist[i,k] * JS_dist[i,k])
                    s_jk = s_jk * s_jk
                    s += s_jk
                #print(s)
                #input()
                u[i,j] = 1 / s
        return u
            
    def update_cluster_center(self, u, glob_iter, selected_users):
        print(u)
        #print(u.shape)
        
        u_p_sum = np.square(u).sum(axis=0)
        #u_test = u.sum(axis=1)
        #print(u_test)
        #input()
        for i in range(self.p):
            for param in self.models[i].parameters():
                param.data = torch.zeros_like(param.data)
            for j, user in enumerate(selected_users):
                self.aggregate_clusterwise(u[:,i], j, user, u_p_sum[i], self.models[i]) #第p簇中心模型更新
                

    def aggregate_clusterwise(self, u_p, j, user, u_p_sum, cluster_model): 
       # weights = {}
        for cluster_param, user_param in zip(cluster_model.parameters(), user.model.parameters()):
            cluster_param.data = cluster_param.data + user_param.data.clone() * u_p[j] * u_p[j]
        cluster_param.data = cluster_param.data / u_p_sum
            
    def get_uij(self,selected_users):
            logits_users = self.get_logits_clients(selected_users)
            logits_centers = self.get_logits_centers()
            u = self.get_u(logits_users, logits_centers)

            return u

    def send_cluster_center(self, u, selected_users):
        for user_idx, user in enumerate(selected_users):
            user.set_uij_parameters(self.models, u, user_idx)
        #for user_idx, user in enumerate(selected_users):
            #cluster_idx = np.argmin(u[user_idx])
            #user.set_parameters(self.models[cluster_idx],beta=1)

    def train(self, args):
        #预训练+初始点选择+初始cluster_assin
        #self.pretrain(args)
        
        cluster_assign_pre = [0] * args.num_users
        cluster_assign_cur = [0] * args.num_users
        last_glob_iter = self.num_glob_iters # kmeans收敛的glob_iter,初始化为最大值
        
        for glob_iter in range(self.num_glob_iters):
            #cluster_assign = []#每个用户对应的类
            print("-------------Round number: ",glob_iter, " -------------")
             #self.send_parameters()    
            self.selected_users = self.select_users(glob_iter,self.num_users)
                
            for user in self.selected_users: # allow selected users to train
                user.compute_weight_update(glob_iter, self.personalized)#loss #计算t轮模型训练结果

            #E步 当前参数估计所属簇
            u = self.get_uij(self.selected_users)
            #self.send_cluster_idx(cluster_assign_cur, self.selected_users)
            # 判断收敛需要好几轮类别都不变才行 这个几轮很难把控。。
            # 全部用户参与时可以只判断两轮一致，part of users参与时必须判断好几轮，但实验证明貌似定不了
            # 因为每轮参与的客户端都是随机的
            #if cluster_assign_pre == cluster_assign_cur: #下一轮训练时不用判断
                #last_glob_iter = glob_iter 
                #print(last_glob_iter,"轮:聚类收敛")
                
            #print(cluster_assign_cur)

            #cluster_assign_pre = cluster_assign_cur 
            
            #M步：用数据更新簇中心
            self.update_cluster_center(u, glob_iter, self.selected_users) #更新类中心模型

            # 发送簇中心
            self.send_cluster_center(u,self.selected_users) # 发送每个用户对应的类中心模型--更新用户模型
            #local_update
            
            
            self.evaluate()

            self.save_results(args)
            #self.save_model_center(args)
            self.save_users_model(args)
            self.save_cluster_assign(args)



            
