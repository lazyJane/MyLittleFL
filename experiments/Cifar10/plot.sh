  python main_plot.py --dataset Cifar10-alpha0.5-ratio0.5-u80 --algorithms  Fedlocal,FedProx,FedAvg,FedKLEM_PerFedAvg,FedKLEM_PerFedAvg_ratio,FedKLEM_PerFedAvg_ratio_e,PerFedavg,FedSEM,FedKLEM,FedIFCA\
  --batch_size 32 --local_epochs 10 --num_users 64 --num_glob_iters 500 --plot_legend 1 --bottom_acc True 

    python main_plot.py --dataset Cifar10-alpha0.5-ratio0.5-u80 --algorithms  Fedlocal,FedProx,FedAvg,FedKLEM_PerFedAvg_ratio,FedKLEM_PerFedAvg_ratio_e,PerFedavg,FedSEM,FedKLEM,FedIFCA\
  --batch_size 32 --local_epochs 10 --num_users 80 --num_glob_iters 500 --plot_legend 1 --bottom_acc True

  

  