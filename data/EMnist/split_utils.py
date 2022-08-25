from xml.dom.minidom import Identified
import numpy as np
import random
import time
import pandas as pd

def rrangebyclass(dataset, n_classes):
    data_indices = list(range(len(dataset)))
    label2index = {k:[] for k in range(n_classes)}
    for idx in data_indices:
        _, label = dataset[idx]
        label2index[label].append(idx)
    return label2index

def iid_divide(l, g):
    """
    https://github.com/TalwalkarLab/leaf/blob/master/data/utils/sample.py
    divide list `l` among `g` groups
    each group has either `int(len(l)/g)` or `int(len(l)/g)+1` elements
    returns a list of groups
    """
    num_elems = len(l)
    group_size = int(len(l) / g) # 每组分到的大小
    num_big_groups = num_elems - g * group_size
    num_small_groups = g - num_big_groups
    glist = []
    for i in range(num_small_groups):
        glist.append(l[group_size * i: group_size * (i + 1)])
    bi = group_size * num_small_groups
    group_size += 1
    for i in range(num_big_groups):
        glist.append(l[bi + group_size * i:bi + group_size * (i + 1)])
    return glist

def split_list_by_indices(l, indices):
    """
    divide list `l` given indices into `len(indices)` sub-lists
    sub-list `i` starts from `indices[i]` and stops at `indices[i+1]`
    returns a list of sub-lists
    """
    res = []
    current_index = 0
    for index in indices:
        res.append(l[current_index: index])
        current_index = index

    return res

def split_iid(indices, n_users, frac, seed=1234):
    rng_seed = (seed if (seed is not None and seed >= 0) else int(time.time())) # 设置种子
    rng = random.Random(rng_seed) # <random.Random object at 0x561bda4e6f30>
    np.random.seed(rng_seed)

    # get subset
    n_samples = int(len(indices) * frac)
    selected_indices = rng.sample(indices, n_samples) # 从list(range(len(dataset)))中无放回随机抽样n_samples条数据【对应的索引】,抽样后list(range(len(dataset)))保持不变
    n_per_users = int(n_samples / n_users) # 取整


    users_indices = [[] for _ in range(n_users)]
    for i in range(n_users):
        users_indices[i] = rng.sample(selected_indices, n_per_users)
        selected_indices = list(set(selected_indices) - set(users_indices[i]))
        
    return users_indices

def pathological_non_iid_split(indices, label2index, n_users, n_classes_per_user, frac, seed=1234):
    rng_seed = (seed if (seed is not None and seed >= 0) else int(time.time()))
    rng = random.Random(rng_seed)
    np.random.seed(rng_seed)

    # get subset
    n_samples = int(len(indices) * frac)
    selected_indices = rng.sample(indices, n_samples)

    sorted_indices = []
    for label in label2index:
        sorted_indices += label2index[label] #健对应的值 # [...]

    n_shards = n_users * n_classes_per_user # 用户数量 * 每个用户的类数 
    shards = iid_divide(sorted_indices, n_shards) # sorted_indices分成n shards 份
    random.shuffle(shards) # 打乱
    user_shards = iid_divide(shards, n_users) # shards分成n_users份

    users_indices = [[] for _ in range(n_users)]
    for user_id in range(n_users):
        for shard in user_shards[user_id]:
            users_indices[user_id] += shard

    return users_indices

def dirichlet_non_iid_split(indices, n_users, dataset, n_classes, alpha, frac, n_clusters=-1, seed=1234):
    if n_clusters == -1:
        n_clusters = n_classes

    rng_seed = (seed if (seed is not None and seed >= 0) else int(time.time()))
    rng = random.Random(rng_seed)
    np.random.seed(rng_seed)

    all_labels = list(range(n_classes)) # [0,1,...,62]
    rng.shuffle(all_labels) # shuffle[0,1,...,62]
    clusters_labels = iid_divide(all_labels, n_clusters) # [[],[],...,[]] 每个簇之间的标签没有重合

    label2cluster = dict()  # maps label to its cluster
    for group_idx, labels in enumerate(clusters_labels): 
        for label in labels: 
            label2cluster[label] = group_idx #标签对应的簇

    # get subset
    n_samples = int(len(indices) * frac)
    selected_indices = rng.sample(indices, n_samples)
    
    # 建立簇与数据索引之间的关系
    clusters_sizes = np.zeros(n_clusters, dtype=int) # [c1_size, c2_size,...cn_size]
    clusters = {k: [] for k in range(n_clusters)} # {c1:[data1,data2,...], c2:[data1,data2,...],...}
    for idx in selected_indices: #簇 
        _, label = dataset[idx] # 这条数据对应的标签
        group_id = label2cluster[label]  #这条数据对应的簇号
        clusters_sizes[group_id] += 1 # 簇size++
        clusters[group_id].append(idx) # 向簇加入这条数据的索引

    for _, cluster in clusters.items(): #_:key  cluster:value
        rng.shuffle(cluster) # 打乱每个簇里的索引号

    users_counts = np.zeros((n_clusters, n_users), dtype=np.int64)  # number of samples by client from each cluster
    #(m，i) 第i个用户来自第m个簇的样本数量

    #np.set_printoptions(threshold=np.inf)
    for cluster_id in range(n_clusters): 
        weights = np.random.dirichlet(alpha=alpha * np.ones(n_users))
        users_counts[cluster_id] = np.random.multinomial(clusters_sizes[cluster_id], weights)  # 对第cluster_id个簇的狄利克雷取样
        
    
    data = pd.DataFrame(users_counts)

    writer = pd.ExcelWriter('A.xlsx')		# 写入Excel文件
    data.to_excel(writer, 'page_1', float_format='%.5f')		# ‘page_1’是写入excel的sheet名
    writer.save()

    writer.close()
   
    '''
    users_counts(i,j) 标签i,用户j
    [[ 11   1   3 ...   9   6  11]
     [ 51 112   0 ... 108  60  30]
     [  4  35  93 ...  30   0  35]
    ...
     [ 80   9 157 ...   1  26  24]
     [  9   0   5 ...  20  15   1]
     [  0   0  17 ...   1   3   3]]
    '''
    users_counts = np.cumsum(users_counts, axis=1)

    users_indices = [[] for _ in range(n_users)]
    for cluster_id in range(n_clusters):
        cluster_split = split_list_by_indices(clusters[cluster_id], users_counts[cluster_id]) # 第i个簇对所有用户的划分
        
        # 将每个用户的样本累加
        for user_id, indices in enumerate(cluster_split):
            #print(indices)
            #input()
            users_indices[user_id] += indices
            #print(users_indices)
            #input()

    return users_indices

def label_swapped_non_iid_split(indices, n_users, frac, seed=1234):
    return split_iid(indices, n_users, frac, seed=1234)