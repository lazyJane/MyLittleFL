from dataclasses import replace
from turtle import clone
from FLAlgorithms.users.userFedKLEM import UserFedKLEM
from FLAlgorithms.servers.serverbase import *
from FLAlgorithms.trainmodel.models_jiu import Net, SimpleLinear
from FLAlgorithms.users.userFedKLEM_PerFedAvg import UserFedKLEM_PerFedAvg
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
class FedKLEM_intra_cluster_PerFedavg(Server):
    def __init__(self, args, model, data_participate, data_unseen, seed):
        super().__init__(args, model, data_participate, data_unseen, seed)

        self.p = args.p
        
        if args.load_model == True:
            self.models = self.load_center_model()
        else: self.models = [create_model_new(args)[0] for p_i in range(self.p)]
        
        for task_id, (train_iterator, val_iterator, test_iterator, len_train, len_test) in \
            enumerate(tqdm(zip(self.train_iterators, self.val_iterators, self.test_iterators, self.len_trains, self.len_tests), total=len(self.train_iterators))):
            if train_iterator is None or test_iterator is None:
                continue
            #GPU_idx = total_users % 6
            #user_device = torch.device("csuda:{}".format(GPU_idx) if torch.cuda.is_available() and args.gpu != -1 else "cpu")
            user = UserFedKLEM_PerFedAvg(args, task_id, model, train_iterator, val_iterator, test_iterator, len_train, len_test, self.len_public, use_adam=False)
            self.users.append(user)
            self.total_train_samples += user.train_samples
        print("Number of users / total users:",args.num_users, " / " , self.total_users)
        print("Finished creating FedKLEM_intra_cluster_PerFedavg server.")

    def load_center_model(self):
        model_path = os.path.join("models", self.dataset)
        for i in range(self.p):
            self.models[i] = torch.load(os.path.join(model_path, "center" + i + ".pt"))
    

    def get_logits_clients(self, users): 
        logits = []
        for i, user in enumerate(users):
            with torch.no_grad():
                for x, y,_ in self.public_loader: 
                    x, y = x.to(self.device), y.to(self.device)
                    #print(y)
                    if 'cifar' in self.dataset or 'shakespeare' in self.dataset: 
                        logits.append(user.model(x))
                    else:
                        logits.append(user.model(x)['logit']) #[tensor(),...tensor()] ???????????????logits
        #print(logits)
        #print(len(logits))
        #input()
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
        #for user in selected_users:
            #for old_param, new_param in zip(user.model.parameters(), self.models[user.cluster_idx].parameters()):
                #old_param.data = new_param.data.clone()
            #user.train(glob_iter)  # ????????????

        local_models = [[] for p in range(self.p)] # ????????????????????????????????????
        for m_i, user in enumerate(selected_users):
            p_i = cluster_assign[m_i] # ???m?????????????????????
            local_models[p_i].append(user.model) 

        for p_i, models in enumerate(local_models):
            if len(models) >0:
                self.aggregate_clusterwise(models, self.models[p_i])

        #self.intra_cluster()
    
    def flatten(self, source):
            return torch.cat([value.flatten() for value in source.values()]) #dim?????????0

    def intra_cluster(self, lamda):
        #delta_w = np.zeros()
        #delta_model = copy.deepcopy(list(self.model.parameters()))
        delta_model = copy.deepcopy(self.model)
        for param in delta_model.parameters():
            param.data = torch.zeros_like(param.data)
        for i,_ in enumerate(self.models):
            for j,_ in enumerate(self.models):
                if i!=j:
                    other_W = {key : value for key, value in self.models[j].named_parameters()}
                    other_W = self.flatten(other_W)
                    for delta_param, other_param in zip(delta_model.parameters(), self.models[j].parameters()):
                        delta_param.data += other_param.data / torch.norm(other_W)
            for param, delta_param in zip(self.models[i].parameters(), delta_model.parameters()):
                param.data = param.data + lamda * delta_param
        return 
            

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
        
        print(centers_indices) #????????????????????????user_idx
        for i in range(self.p):
            self.models[i] = copy.deepcopy(self.users[centers_indices[i]].model)

    def pretrain(self, args):
        pre_train_times = 10
        #??????
        self.send_parameters()    
        for user in self.users: 
            for i in range(pre_train_times):
                user.train(i) 
        
        self.get_logits_clients(self.users)
        centers = self.center_init_kmeansplus() #???????????????

    def update_unseen_users(self, unseen_E):
        #???????????????
        self.unseen_user_train(unseen_E)
        #??????????????????????????????
        cluster_assign_unseen = self.get_cluster(self.unseen_users)
        for i, user in enumerate(self.unseen_users):
            cluster_idx = cluster_assign_unseen[i]
            user.set_parameters(self.models[cluster_idx])

    def train(self, args):
        #?????????+???????????????+??????cluster_assin
        #self.pretrain(args)
        
        cluster_assign_pre = [0] * args.num_users
        cluster_assign_cur = [0] * args.num_users
        last_glob_iter = self.num_glob_iters # kmeans?????????glob_iter,?????????????????????
        
        for glob_iter in range(self.num_glob_iters):
            #cluster_assign = []#????????????????????????
            print("-------------Round number: ",glob_iter, " -------------")
            
            self.selected_users = self.select_users(glob_iter,self.num_users)
                
            for user in self.selected_users: # allow selected users to train
                user.train(glob_iter, self.personalized)#loss #??????t?????????????????????

            self.save_users_model(args)
            #E??? ???????????????????????????
            cluster_assign_cur = self.get_cluster(self.selected_users)
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
            self.update_cluster_center(cluster_assign_cur, glob_iter, self.selected_users) #?????????????????????
            self.intra_cluster(self.beta)

            # ???????????????
            self.send_cluster_center() # ??????????????????????????????????????????--??????????????????
            #local_update
            
            
            self.evaluate()

            self.save_results(args)
            self.save_model_center(args)
            
            self.save_cluster_assign(args)
        self.update_unseen_users(unseen_E = 5)
        self.evaluate_unseen_users()





            
