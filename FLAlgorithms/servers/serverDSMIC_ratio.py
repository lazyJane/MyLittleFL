from dataclasses import replace
from turtle import clone
from FLAlgorithms.users.userFedKLEM_PerFedAvg import UserFedKLEM_PerFedAvg
from FLAlgorithms.servers.serverbase import *
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
import copy
import torch
class FedKLEM_PerFedAvg_ratio(Server):
    def __init__(self, args, model, data_participate, data_unseen, seed):
        super().__init__(args, model, data_participate, data_unseen, seed)

        self.p = args.p

        if args.load_model == True:
                self.models = self.load_center_model()
        else: self.models = [create_model_new(args)[0] for p_i in range(self.p)]

        if args.test_unseen == True:
            a = 1
        else:
            for task_id, (train_iterator, val_iterator, test_iterator, len_train, len_test) in \
                enumerate(tqdm(zip(self.train_iterators, self.val_iterators, self.test_iterators, self.len_trains, self.len_tests), total=len(self.train_iterators))):
                if train_iterator is None or test_iterator is None:
                    continue
                #GPU_idx = total_users % 6
                #user_device = torch.device("cuda:{}".format(GPU_idx) if torch.cuda.is_available() and args.gpu != -1 else "cpu")
                user = UserFedKLEM_PerFedAvg(args, task_id, model, train_iterator, val_iterator, test_iterator, len_train, len_test, self.len_public, use_adam=False)
                self.users.append(user)
                self.total_train_samples += user.train_samples
            print("Number of users / total users:",args.num_users, " / " , self.total_users)
            print("Finished creating FedKLEM_PerFedAvg server.")
    

    def get_logits_clients(self, users): 
        logits = []
        for i, user in enumerate(users):
            with torch.no_grad():
                for x, y,_ in self.public_loader: 
                    x, y = x.to(self.device), y.to(self.device)
                    if 'cifar' in self.dataset or 'shakespeare' in self.dataset: 
                        logits.append(user.model(x))
                    else:
                        logits.append(user.model(x)['logit']) #[tensor(),...tensor()] ???????????????logits
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

    def get_cluster_idx(self, logits_users, logits_centers): #[tensor(),...tensor()]
        cluster_assign = []
        for i, logits_i in enumerate(logits_users):
            angles = torch.zeros(self.p)
            for j, logits_j in enumerate(logits_centers):
        
                angles[j] = F.kl_div(F.log_softmax(logits_i, dim=1), F.softmax(logits_j, dim=1), reduction="batchmean")
            #print(angles)
            min_p_i = np.argmin(angles.numpy()) # ???i???????????????j??????
            #print(min_p_i)
            cluster_assign.append(min_p_i)
        
        return cluster_assign

    def update_cluster_center(self, cluster_assign, glob_iter, selected_users):
        local_models = [[] for p in range(self.p)] # ????????????????????????????????????
        local_users = [[] for p in range(self.p)]
        for m_i, user in enumerate(selected_users):
            p_i = cluster_assign[m_i] # ???m?????????????????????
            local_models[p_i].append(user.model) 
            local_users[p_i].append(user)

        self.timestamp = time.time()
        for p_i, users in enumerate(local_users):
            if len(users) >0:
                self.aggregate_clusterwise(users, self.models[p_i])
        curr_timestamp=time.time() 
        agg_time = curr_timestamp - self.timestamp

        return agg_time

    def add_parameters(self, user, global_model, ratio):
       for server_param, user_param in zip(global_model.parameters(), user.model.parameters()):
            server_param.data = server_param.data + user_param.data.clone() * ratio

    def aggregate_clusterwise(self, local_users, global_model): 
        for param in global_model.parameters():
            param.data = torch.zeros_like(param.data) # cluster_center?????????????????????0
        total_train = 0
        for user in local_users:
            total_train += user.train_samples # ????????????????????????
        for m_i, user in enumerate(local_users):
            self.add_parameters(user, global_model, user.train_samples / total_train)
    '''
    def aggregate_clusterwise(self, local_models, global_model): 
        weights = {}

        for m_i, local_model in enumerate(local_models):
            for name, param in local_model.named_parameters():
                if name not in weights:
                    weights[name] = torch.zeros_like(param.data)

                weights[name] += param.data

        for name, param in global_model.named_parameters(): # ?????????????????????
            weights[name] /= len(local_models)
            param.data = weights[name]
    '''
            
    def get_cluster(self, selected_users):
        logits_users = self.get_logits_clients(selected_users)
        logits_centers = self.get_logits_centers()
        cluster_assign = self.get_cluster_idx(logits_users, logits_centers)

        return cluster_assign

    def send_cluster_idx(self, cluster_assign, selected_users):
        for i, user in enumerate(selected_users):
            user.set_cluster_idx(cluster_assign[i])
    
    def send_cluster_center(self, beta=1):
        for user in self.users:
            cluster_idx = user.cluster_idx
            user.set_parameters(self.models[cluster_idx],beta=beta)

    def center_init(self): # k-means????????????,?????????user?????????
        user_indices = list(range(len(self.user)))
        centers_indices = [] #?????????????????????
        rng = random.Random(5)
        for i in range(self.p):
            centers_indices.append(rng.sample(user_indices, 1))
            user_indices = list(set(user_indices) - set(centers_indices))
        for i in range(self.p):
            self.models[i] = copy.deepcopy(self.user[centers_indices[i]].model)

    def closet_dist(self, user, center_users): #???????????????????????????????????????????????????
        user_to_list = []
        user_to_list.append(user)
        logits_users_j = self.get_logits_clients(user_to_list)[0]
        
        logits_centers = self.get_logits_clients(center_users)
        min_dist = float('inf')
        for logits in logits_centers:
            dist = F.kl_div(F.log_softmax(logits_users_j, dim=1), F.softmax(logits, dim=1), reduction="batchmean")
            if dist < min_dist:
                min_dist = dist 
        return min_dist


    def center_init_kmeansplus(self):
        rng = random.Random(5)
    
        user_indices = list(range(len(self.users)))
        centers_indices = [] #?????????????????????
        centers_indices.append(rng.sample(user_indices, 1)[0]) #??????????????????????????????????????????
        #print(centers_indices)
        dist = [0 for i in range(len(self.users))] #???????????????????????????????????????????????????????????????
        for i in range(self.p-1):
            dist_sum = 0
            for j, user in enumerate(self.users):
                center_users = [self.users[i] for i in centers_indices]
                dist[j] = self.closet_dist(user, center_users)
            dist_sum += dist[j]
            dist_sum *= random.random() #random.random ????????????0-1???????????????
            #???dist??????????????????????????????????????????
            #?????????????????????????????????
            for j, dist_j in enumerate(dist):
                dist_sum -= dist_j
                if dist_sum > 0 :
                    continue
                #dist_sum<0
                centers_indices.append(user_indices[j])
                break
        
        print(centers_indices)
        for i in range(self.p):
            self.models[i] = copy.deepcopy(self.users[centers_indices[i]].model)

    def pretrain(self, args):
        pre_train_times = 20
        #??????
        for user in self.users: 
            for i in range(pre_train_times):
                user.train(i) 
               
        centers = self.center_init_kmeansplus() #???????????????

        for user in self.users: #????????????????????????????????????????????????????????????
            cluster_assign_initial = self.get_cluster(self.users)
            self.send_cluster_idx(cluster_assign_initial, self.users) #??????????????????
        self.save_cluster_assign(args)

    def evaluate_one_step(self, save=True):
        for c in self.users:
            c.train_one_step()

        self.evaluate()

        # set local model back to client for training process.
        for c in self.users:
            c.update_parameters(c.local_model)

    def update_unseen_users(self, unseen_E = 0, unseen_local_epochs = 0):
        #???????????????s
        self.unseen_user_train(unseen_E, unseen_local_epochs)
        #??????????????????????????????
        cluster_assign_unseen = self.get_cluster(self.unseen_users)
        print(cluster_assign_unseen)
        for i, user in enumerate(self.unseen_users):
            cluster_idx = cluster_assign_unseen[i]
            user.set_parameters(self.models[cluster_idx])
        
        #for glob_iter in range(10):
            #for user in self.unseen_users:
                #user.train_unseen(glob_iter)
        

    def train(self, args):
        #?????????+???????????????+??????cluster_assin
        #self.pretrain(args)
        
        cluster_assign_pre = [0] * args.num_users
        cluster_assign_cur = [0] * args.num_users
        last_glob_iter = self.num_glob_iters # kmeans?????????glob_iter,?????????????????????
        user_time = 0
        server_time = 0
        train_start= time.time() 
        
        for glob_iter in range(self.num_glob_iters):
            #cluster_assign = []#????????????????????????
            print("-------------Round number: ",glob_iter, " -------------")
             #self.send_parameters()    
            self.selected_users = self.select_users(glob_iter,self.num_users)
            
            self.timestamp = time.time() 
            for user in self.selected_users: # allow selected users to train
                user.train(glob_iter, self.personalized)#loss #??????t?????????????????????
            curr_timestamp = time.time()
            train_time = (curr_timestamp - self.timestamp) / len(self.selected_users)
            self.metrics['user_train_time'].append(train_time)
            user_time += train_time 

            self.evaluate_one_step()

            self.save_users_model(args)

            self.timestamp = time.time()
            #E??? ???????????????????????????
            cluster_assign_cur = self.get_cluster(self.selected_users)
            curr_timestamp=time.time() 
            cluster_time = curr_timestamp - self.timestamp

            self.send_cluster_idx(cluster_assign_cur, self.selected_users)
            # ???????????????????????????????????????????????? ??????????????????????????????
            # ???????????????????????????????????????????????????part of users???????????????????????????????????????????????????????????????
            # ?????????????????????????????????????????????
            if cluster_assign_pre == cluster_assign_cur: #??????????????????????????????
                last_glob_iter = glob_iter 
                print(last_glob_iter,"???:????????????")
                
            print(cluster_assign_cur)

            cluster_assign_pre = cluster_assign_cur 
            
            #M??????????????????????????????
            agg_time = self.update_cluster_center(cluster_assign_cur, glob_iter, self.selected_users) #?????????????????????
            server_total_time = cluster_time + agg_time
            self.metrics['server_agg_time'].append(server_total_time)
            server_time += server_total_time

            # ???????????????
            self.send_cluster_center() # ??????????????????????????????????????????--??????????????????
            #local_update
            
            
            #self.evaluate()

            self.save_results(args)
            self.save_model_center(args)
            
            self.save_cluster_assign(args)

        train_end = time.time()
        total_train_time = train_end - train_start
        self.metrics['total_train_time'].append(total_train_time)
        print("user_time = {:.4f}, server_time = {:.4f}, total_time = {:.4f}.".format(user_time, server_time, total_train_time))
        #self.update_unseen_users(unseen_E = 5)
        #self.evaluate_unseen_users()





            
