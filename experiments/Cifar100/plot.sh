
      python main_plot.py --dataset Cifar100-alpha0.5-ratio0.5-u125 --algorithms  Fedlocal,FedAvg,FedProx,FedIFCA,FedSEM,PerFedavg,FedKLEM_PerFedAvg,FedKLEM_PerFedAvg_ratio,FedKLEM_PerFedAvg_ratio_e,FedKLEM\
  --batch_size 32 --E 3 --num_users 100 --num_glob_iters 500 --plot_legend 1 --bottom_acc True --times 3

  
      python main_plot.py --dataset Cifar100-alpha0.5-ratio0.5-u125 --algorithms  FedIFCA,FedSEM,PerFedavg,FedKLEM_PerFedAvg,FedKLEM_PerFedAvg_ratio,FedKLEM_PerFedAvg_ratio_e,FedKLEM\
  --batch_size 32 --E 6 --num_users 100 --num_glob_iters 500 --plot_legend 1 --bottom_acc True --time 3

  