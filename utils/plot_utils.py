from re import A
import matplotlib.pyplot as plt
#from pyrsistent import v
import h5py
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
from matplotlib.ticker import StrMethodFormatter
import os
from utils.model_utils import get_log_path, METRICS
import seaborn as sns
import string
import matplotlib.colors as mcolors
import pandas as pd
import os
COLORS=list(mcolors.TABLEAU_COLORS)
MARKERS=["o", "v", "s", "*", "x", "P","4","|","h","d","X","p"]

plt.rcParams.update({'font.size': 14})
n_seeds=3

def load_results(args, algorithm, seed):
    alg = get_log_path(args, algorithm, seed, args.gen_batch_size)

    hf = h5py.File("./{}/{}.h5".format(args.result_path, alg), 'r')
    metrics = {}
    for key in METRICS:
        metrics[key] = np.array(hf.get(key)[:])
    return metrics


def get_label_name(name):
    #name = name.split("_")[0]
    if 'FedKLEM_PerFedAvg_ratio_e' in name:
        name = 'ratio_e' 
    elif 'FedKLEM_PerFedAvg_ratio' in name:
        name = 'ratio'
    elif 'FedKLEM_intra_cluster_PerFedavg' in name:
        name = 'FedKLEM_ic_PerFedavg' 
    elif 'FedKLEM_IntraC' in name:
        name = 'FedKLEM_IntraC' 
    elif 'KLEM_PerFedAvg' in name:
        name = 'FedKLEM_PerFedAvg' 
    elif 'KLEM' in name:
        name = 'FedKLEM'
    elif 'FedDF' in name:
        name = 'FedFusion'
    elif 'FedEnsemble' in name:
        name = 'Ensemble'
    elif 'PerFedAvg' in name:
        name = 'PerFedAvg'
    elif 'FedAvg' in name:
        name = 'FedAvg'
    return name

def plot_results(args, algorithms):
    if args.unseen_test_acc:
        args.result_path = args.result_path + "unseen"
    n_seeds = args.times 
    if 'femnist' in args.dataset.lower() or 'shakespeare' in args.dataset.lower(): #  FEMNIST-alpha0.4-ratio0.5
        sub_dir = args.dataset
    else: #  Mnist-alpha0.4-ratio0.5
        dataset_ = args.dataset.split('-') 
        sub_dir = dataset_[0] + "/" +  dataset_[1] + "/" + dataset_[2] # e.g. Mnist/ratio0.5
        #
    '''
    'unseen_train_acc',
    'unseen_train_loss',
    'unseen_glob_acc',#test
    'unseen_glob_loss',
    'unseen_glob_acc_bottom']
    '''
    #if args.unseen_train_acc:
        #show = 'unseen_train_acc'
        #sub_dir += '/unseen_train_acc'
    if args.unseen_test_acc:
        show = 'unseen_glob_acc'
        sub_dir += '/unseen_glob_acc'
    if args.train_acc:
        show = 'train_acc'
        sub_dir += '/train_acc'
    elif args.train_loss:
        show = 'train_loss'
        sub_dir += '/train_loss'
    elif args.test_acc:
        show = 'glob_acc'
        sub_dir += '/test_acc'
    elif args.test_loss:
        show = 'glob_loss'
        sub_dir += '/test_loss'
    elif args.bottom_acc:
        show = 'glob_acc_bottom_2'
        show1 = 'glob_acc_bottom_1'
        show2 = 'glob_acc_bottom_2'
        show3 = 'glob_acc_bottom_3'
        show4 = 'glob_acc_bottom_4'
        show5 = 'glob_acc_bottom_5'
        sub_dir += '/bottom_acc'
    elif args.bottom_acc_1:
        show = 'glob_acc_bottom_1'
        sub_dir += '/bottom_acc'
    elif args.bottom_acc_2:
        show = 'glob_acc_bottom_2'
        sub_dir += '/bottom_acc'
    elif args.bottom_acc_3:
        show = 'glob_acc_bottom_3'
        sub_dir += '/bottom_acc'
    elif args.bottom_acc_4:
        show = 'glob_acc_bottom_4'
        sub_dir += '/bottom_acc'
    elif args.bottom_acc_5:
        show = 'glob_acc_bottom_5'
        sub_dir += '/bottom_acc'
    os.system("mkdir -p figs/{}".format(sub_dir))  # e.g. figs/Mnist/ratio0.5
    plt.figure(1, figsize=(5, 5))
    TOP_N = 5
    max_acc = 0
    all_metrics = {algorithm:[] for algorithm in algorithms}

    for i, algorithm in enumerate(algorithms):
        algo_name = get_label_name(algorithm)
        metrics = [load_results(args, algorithm, seed) for seed in range(n_seeds)]
        #print(metrics)
        #print(metrics[0][show])
        #for i in metrics[0][show]:
        
        all_metrics[algorithm]=metrics[0][show]

        ######### plot test accuracy ############
       
        all_curves = np.concatenate([metrics[seed][show] for seed in range(n_seeds)])
        print(len(all_curves))
        #input()
        if args.bottom_acc:
            top_acc1 = np.concatenate([np.sort(metrics[seed][show1])[-TOP_N:] for seed in range(n_seeds)])
            top_acc2 = np.concatenate([np.sort(metrics[seed][show2])[-TOP_N:] for seed in range(n_seeds)])
            top_acc3 = np.concatenate([np.sort(metrics[seed][show3])[-TOP_N:] for seed in range(n_seeds)])
            top_acc4 = np.concatenate([np.sort(metrics[seed][show4])[-TOP_N:] for seed in range(n_seeds)])
            top_acc5 = np.concatenate([np.sort(metrics[seed][show5])[-TOP_N:] for seed in range(n_seeds)])
            acc_avg1 = np.mean(top_acc1)
            acc_avg2 = np.mean(top_acc2)
            acc_avg3 = np.mean(top_acc3)
            acc_avg4 = np.mean(top_acc4)
            acc_avg5 = np.mean(top_acc5)
            acc_avg = (acc_avg1 + acc_avg2 + acc_avg3 + acc_avg4 + acc_avg5) / 5.00
            acc_std = np.std(top_acc1)

        else:
            top_accs =  np.concatenate([np.sort(metrics[seed][show])[-TOP_N:] for seed in range(n_seeds)])
        
            acc_avg = np.mean(top_accs)
            acc_std = np.std(top_accs)

        info = 'Algorithm: {:<10s}, Accuracy = {:.2f} %, deviation = {:.2f}'.format(algo_name, acc_avg * 100, acc_std * 100)
        print(info)
        length = len(all_curves) // n_seeds
        #print()
        
        sns.lineplot(
            x=np.array(list(range(length)) * n_seeds) + 1,
            y=all_curves.astype(float),
            legend='brief',
            #marker=MARKERS[i],
            color=COLORS[i],
            label=algo_name,
            ci="sd",
        )
        
    print(len(all_metrics))
   
    plt.gcf()
    plt.grid()
    #os.system("mkdir -p figs/{}".format(sub_dir)) 
    if 'femnist' in args.dataset.lower() or 'shakespeare' in args.dataset.lower():
        dataset_ = args.dataset
        if args.E > 0:
                fig_save_path = os.path.join('figs', sub_dir, dataset_[0]  + '_E' + str(args.E) + '_user' + str(args.num_users) + '.png')
        if args.local_epochs > 0:
                fig_save_path = os.path.join('figs', sub_dir, dataset_[0]  + '_local_epochs' + str(args.local_epochs) + '_user' + str(args.num_users) + '.png')
        prefix_all_metrics_path = os.path.join('figs', sub_dir, dataset_[0] + dataset_[1])
    else:
        dataset_ = args.dataset.split('-')#['Mnist', 'alpha0.5', 'ratio1.0', 'u100']
        if 'alpha' in args.dataset:
            if args.E > 0:
                fig_save_path = os.path.join('figs', sub_dir, dataset_[0] + '-' + dataset_[2] + '_E' + str(args.E) + '_user' + str(args.num_users) + '.png')
            if args.local_epochs > 0:
                fig_save_path = os.path.join('figs', sub_dir, dataset_[0] + '-' + dataset_[2] + '_local_epochs' + str(args.local_epochs) + '_user' + str(args.num_users) + '.png')
            prefix_all_metrics_path = os.path.join('figs', sub_dir, dataset_[0] + '-' + dataset_[1])
        else: 
            fig_save_path = os.path.join('figs', sub_dir, dataset_[0] + '-' + dataset_[1] + '.png')
            prefix_all_metrics_path = os.path.join('figs', sub_dir, dataset_[0] + '-' + dataset_[1])
    
    if args.E > 0:
        all_metrics_path = os.path.join(prefix_all_metrics_path + "E = " + str(args.E) + '.all_metrics.xlsx')
    elif args.local_epochs > 0:
        all_metrics_path = os.path.join(prefix_all_metrics_path +  "local_epochs = " + str(args.local_epochs) + '.all_metrics.xlsx')

    plt.title(dataset_[0] + ' Test Accuracy')
    plt.xlabel('Epoch')
    max_acc = np.max([max_acc, np.max(all_curves) ]) + 4e-2



    #if args.min_acc < 0:
    #    alpha = 0
    #    min_acc = np.max(all_curves) * alpha + np.min(all_curves) * (1-alpha)
    #else:
    #    min_acc = args.min_acc
    #plt.ylim(min_acc, max_acc)
    
    #fig_save_path = os.path.join('figs', sub_dir, dataset_[0] + '-' + dataset_[2] + '.png')
    plt.savefig(fig_save_path, bbox_inches='tight', pad_inches=0, format='png', dpi=400)
    print('file saved to {}'.format(fig_save_path))

    print(all_metrics_path)
    for value in all_metrics.items():
        print(len(value[1]))
    data = pd.DataFrame(all_metrics)

    writer = pd.ExcelWriter(all_metrics_path)		# 写入Excel文件
    data.to_excel(writer, 'page_1', float_format='%.5f')		# ‘page_1’是写入excel的sheet名
    writer.save()

    writer.close()