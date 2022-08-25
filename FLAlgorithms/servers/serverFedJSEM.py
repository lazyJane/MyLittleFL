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

class FedJSEM(Server):
    def __init__(self, args, model, data_participate, data_unseen, seed):
        super().__init__(args, model, data_participate, data_unseen, seed)

        self.p = args.p
        self.models = [create_model_new(args)[0] for p_i in range(self.p)]

        for task_id, (train_iterator, val_iterator, test_iterator, len_train, len_test) in \
            enumerate(tqdm(zip(self.train_iterators, self.val_iterators, self.test_iterators, self.len_trains, self.len_tests), total=len(self.train_iterators))):
            if train_iterator is None or test_iterator is None:
                continue
            user = UserFedKLEM(args, task_id, model, train_iterator, val_iterator, test_iterator, len_train, len_test, self.len_public, use_adam=False)
            self.users.append(user)
            self.total_train_samples += user.train_samples
        print("Number of users / total users:",args.num_users, " / " , self.total_users)
        print("Finished creating FedKLEM server.")



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
                            logits.append(self.models[i](x)) # torch.Size([32, 100, 80])
                        else:
                            logits.append(self.models[i](x)['logit']) #[tensor(),...tensor()]
        return logits    

    def get_cluster_idx(self, logits_users, logits_centers): #[tensor(),...tensor()]
        cluster_assign = []
        for i, logits_i in enumerate(logits_users):
            angles = torch.zeros(self.p)
            for j, logits_j in enumerate(logits_centers):
                p_output = F.softmax(logits_i, dim=1)
                q_output = F.softmax(logits_j, dim=1)
                log_mean_output = ((p_output + q_output )/2).log()
                angles[j] = (F.kl_div(log_mean_output, p_output) + F.kl_div(log_mean_output, q_output)) / 2
            min_p_i = np.argmin(angles.numpy()) # 第i个用户属于j个簇
            #print(min_p_i)
            cluster_assign.append(min_p_i)
        
        return cluster_assign

    def update_cluster_center(self, cluster_assign, glob_iter, selected_users):
        #for user in selected_users:
            #for old_param, new_param in zip(user.model.parameters(), self.models[user.cluster_idx].parameters()):
                #old_param.data = new_param.data.clone()
            #user.train(glob_iter)  # 用户模型

        local_models = [[] for p in range(self.p)] # 存储每个类对应的用户模型
        for m_i, user in enumerate(selected_users):
            p_i = cluster_assign[m_i] # 第m个用户对应的类
            local_models[p_i].append(user.model) 

        for p_i, models in enumerate(local_models):
            if len(models) >0:
                self.aggregate_clusterwise(models, self.models[p_i])
            

    def aggregate_clusterwise(self, local_models, global_model): 
        weights = {}

        for m_i, local_model in enumerate(local_models):
            for name, param in local_model.named_parameters():
                if name not in weights:
                    weights[name] = torch.zeros_like(param.data)

                weights[name] += param.data

        for name, param in global_model.named_parameters(): # 更新类中心模型
            weights[name] /= len(local_models)
            param.data = weights[name]

            
    def get_cluster(self, selected_users):
        logits_users = self.get_logits_clients(selected_users)
        logits_centers = self.get_logits_centers()
        cluster_assign = self.get_cluster_idx(logits_users, logits_centers)

        return cluster_assign

    def send_cluster_idx(self, cluster_assign, selected_users):
        for i, user in enumerate(selected_users):
            user.set_cluster_idx(cluster_assign[i])


    def save_model_center(self, args):
        model_path = os.path.join("models", self.dataset, args.algorithm)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        for i in range(self.p):
            torch.save(self.models[i], os.path.join(model_path, "center" + str(i) + ".pt"))
    
    def send_cluster_center(self, users, beta=1):
        for user in users:
            cluster_idx = user.cluster_idx
            user.set_parameters(self.models[cluster_idx],beta=beta)


    def train(self, args):
        cluster_assign_pre = [0] * args.num_users
        cluster_assign_cur = [0] * args.num_users
        last_glob_iter = self.num_glob_iters # kmeans收敛的glob_iter,初始化为最大值
        for glob_iter in range(self.num_glob_iters):
            #cluster_assign = []#每个用户对应的类
            print("-------------Round number: ",glob_iter, " -------------")
            #self.send_p_parameters(self.models) #发送p个模型参数
            self.selected_users = self.select_users(glob_iter,self.num_users)
    
            #if glob_iter <= last_glob_iter:
            cluster_assign_cur = self.get_cluster(self.selected_users)
            self.send_cluster_idx(cluster_assign_cur, self.selected_users)
            # 判断收敛需要好几轮类别都不变才行 这个几轮很难把控。。
            # 全部用户参与时可以只判断两轮一致，part of users参与时必须判断好几轮，但实验证明貌似定不了
            # 因为每轮参与的客户端都是随机的
            if cluster_assign_pre == cluster_assign_cur: #下一轮训练时不用判断
                last_glob_iter = glob_iter 
                print(last_glob_iter,"轮:聚类收敛")
                
            
            print(cluster_assign_cur)

            cluster_assign_pre = cluster_assign_cur 
            
            self.update_cluster_center(cluster_assign_cur, glob_iter, self.selected_users) #更新类中心模型
            self.send_cluster_center(self.users) # 发送每个用户对应的类中心模型--更新用户模型
            
            for user in self.selected_users: # allow selected users to train
                user.compute_weight_update(glob_iter, self.personalized)#loss

            self.evaluate()

            self.save_results(args)
            self.save_model_center(args)
            self.save_users_model(args)
            self.save_cluster_assign(args)

    

    




            
