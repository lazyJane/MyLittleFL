from FLAlgorithms.users.userFedKL import UserFedKL
from FLAlgorithms.servers.serverbase import Server
#from utils import public_data
import numpy as np
# Implementation for FedCluster Server
import time
import torch 
from sklearn.cluster import AgglomerativeClustering
from utils.helper import ExperimentLogger, display_train_stats
import math
import matplotlib.pyplot as plt
import torch.nn.functional as F

class FedKL(Server):
    def __init__(self, args, model, seed):
        super().__init__(args, model, seed)

        # Initialize data for all users
        data = read_data(args)#data:users, groups, train_data, test_data, proxy_data, len   train_data, test_data: a dict{'user':x,y}

        self.public_loader , self.iter_proxyloader = read_proxy_data(data[4],'',data[5])
        
        total_users = len(data[0]) #总用户数
        print("Users in total: {}".format(total_users))
        
        self.p = 2 # 数据rotated分类数
        m_per_cluster = total_users // self.p
        cluster_assign_data = []
        #for p_i in range(self.p):
            #cluster_assign_data += [p_i for _ in range(m_per_cluster)]#用户编号对应的类别编号
        
        #### creating users ####
        for i in range(total_users):
            #p_i = cluster_assign_data[i]
            #id, train_data , test_data = read_user_data_Rotated(i, self.p, p_i, data, dataset=args.dataset)
            id, train_data , test_data = read_user_data(i, data, dataset=args.dataset)
            user = UserFedKL(args, id, model, train_data, test_data, use_adam=False)
            self.users.append(user) # 加入用户
            self.total_train_samples += user.train_samples

        print("Number of users / total users:",args.num_users, " / " ,total_users)
        print("Finished creating FedCluster server.")

        self.cluster_indices = [np.arange(self.num_users).astype("int")] 
        self.user_clusters = [[self.users[i] for i in idcs] for idcs in self.cluster_indices]

    def flatten(self, source):
            return torch.cat([value.flatten() for value in source.values()]) #dim默认为0

    def get_logits_clients(self, users):
            logits = []
            for i, user in enumerate(self.users):
                with torch.no_grad():
                    for x, y in self.public_loader: 
                        x, y = x.to(self.device), y.to(self.device)
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

            return angles.numpy()
    
    def compute_pairwise_JS(self, sources):
            angles = torch.zeros([len(sources), len(sources)])
            for i, source1 in enumerate(sources):
                 for j, source2 in enumerate(sources):
                    p_output = F.softmax(source1, dim=1)
                    q_output = F.softmax(source2, dim=1)
                    log_mean_output = ((p_output + q_output )/2).log()
                    angles[i,j] = (F.kl_div(log_mean_output, p_output) + F.kl_div(log_mean_output, q_output)) / 2
            #print(angles)
            #input()
            return angles.numpy()
        
    def cluster_users(self, S):
        clustering = AgglomerativeClustering(affinity="precomputed", linkage="complete").fit(S)

        c1 = np.argwhere(clustering.labels_ == 0).flatten() 
        c2 = np.argwhere(clustering.labels_ == 1).flatten() 
        return c1, c2

    def reduce_add_average(self, targets, sources):
        for target in targets:
            for name in target:
                tmp = torch.mean(torch.stack([source[name].data for source in sources]), dim=0).clone() #所有用户模型参数的均值
                target[name].data += tmp #reset之后每个类内客户端的参数跟训练之前保持一致，所以一个类内每个客户端w+mean（▲W）都相等

    def aggregate_clusterwise(self, user_clusters):
        for cluster in user_clusters:
            self.reduce_add_average(targets=[user.W for user in cluster], 
                               sources=[user.dW for user in cluster], )



    def train(self, args):
        for glob_iter in range(self.num_glob_iters):
            print("\n\n-------------Round number: ",glob_iter, " -------------\n\n")
            self.selected_users = self.select_users(glob_iter,self.num_users)
            if glob_iter == 1:
                self.send_parameters(mode=self.mode)
            self.evaluate() # 每个用户在自己的测试集上测试准确率的均值 

            self.timestamp = time.time() # log user-training + compute dW + similarities start time train+计算dW时间
            for user in self.selected_users: # allow selected users to train
                user.compute_weight_update(glob_iter, self.personalized)#loss
                user.reset()
            logits = self.get_logits_clients(self.selected_users)
            similarities = self.compute_pairwise_JS(logits)##shape[10,10] 计算相似度
            curr_timestamp = time.time() # log user-training + compute dW + similarities start time
            compute_cos_time = (curr_timestamp - self.timestamp) / len(self.selected_users)
            #self.metrics['compute_cos_time'].append(compute_cos_time)
            #print('compute_cos_time',compute_cos_time)
                
            cluster_indices_new = []
            for idc in self.cluster_indices: #idc:[1,2,...,10]  对每个类
                if  len(idc)>3 and glob_iter>0: #0.4, 1.6
                    print("执行了一次聚类")
                    #server.cache_model(idc, users[idc[0]].W, acc_users) 
                    c1, c2 = self.cluster_users(similarities[idc][:,idc]) #[[1,2,...,10]][:,[1,2,...,10]]取这一类的相似度矩阵

                    cluster_indices_new += [c1, c2]
                    print(cluster_indices_new)

                    print("split",glob_iter)

                else:
                    cluster_indices_new += [idc]

            self.cluster_indices = cluster_indices_new # 新的类划分
            self.user_clusters = [[self.users[i] for i in idcs] for idcs in self.cluster_indices]

            self.timestamp = time.time() # log server-agg start time
            self.aggregate_clusterwise(self.user_clusters)
            curr_timestamp=time.time()  # log  server-agg end time
            agg_time = curr_timestamp - self.timestamp
            self.metrics['server_agg_time'].append(agg_time)
            #print('server_agg_time',agg_time)
            

            self.save_results(args)
            #self.save_model()