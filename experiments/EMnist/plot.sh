  python main_plot.py --dataset EMnist-alpha1.0-ratio1.0-u100 --algorithms Fedlocal,FedAvg,FedSEM,FedProx,FedIFCA,FedKLEM,PerFedavg,FedKLEM_PerFedAvg,FedKLEM_PerFedAvg_ratio,FedKLEM_PerFedAvg_ratio_e\
  --batch_size 32 --local_epochs 10 --num_users 80 --num_glob_iters 200 --plot_legend 1 --bottom_acc True

    python main_plot.py --dataset EMnist-alpha0.5-ratio1.0-u100 --algorithms FedIFCA,FedKLEM,PerFedavg,FedKLEM_PerFedAvg,FedKLEM_PerFedAvg_ratio,FedKLEM_PerFedAvg_ratio_e\
  --batch_size 32 --local_epochs 10 --num_users 40 --num_glob_iters 200 --plot_legend 1 --test_acc True

  python main_plot.py --dataset EMnist-alpha0.5-ratio1.0-u100 --algorithms Fedlocal,FedKLEM,FedIFCA,PerFedavg,FedKLEM_PerFedAvg,FedKLEM_PerFedAvg_ratio,FedKLEM_PerFedAvg_ratio_e,FedSEM\
  --batch_size 32 --local_epochs 20 --num_users 80 --num_glob_iters 200 --plot_legend 1 --test_acc True

    python main_plot.py --dataset EMnist-alpha10.0-ratio1.0-u100 --algorithms FedAvg,FedSEM,Fedlocal,FedProx,FedIFCA,FedKLEM,PerFedavg,FedKLEM_PerFedAvg,FedKLEM_PerFedAvg_ratio,FedKLEM_PerFedAvg_ratio_e\
  --batch_size 32 --local_epochs 10 --num_users 80 --num_glob_iters 200 --plot_legend 1 --bottom_acc True --times 3

    python main_plot.py --dataset EMnist-alpha0.05-ratio1.0-u100 --algorithms PerFedavg,FedKLEM_PerFedAvg_ratio,FedKLEM_PerFedAvg_ratio_e,Fedlocal,FedProx,FedKLEM,FedAvg,FedIFCA,FedSEM \
  --batch_size 32 --local_epochs 10 --num_users 36 --num_glob_iters 200 --plot_legend 1 --test_acc True

  python main_plot.py --dataset EMnist-alpha5.0-ratio1.0-u100 --algorithms FedIFCA,FedKLEM,PerFedavg,FedKLEM_PerFedAvg,FedKLEM_PerFedAvg_ratio,FedKLEM_PerFedAvg_ratio_e\
  --batch_size 32 --local_epochs 10 --num_users 80 --num_glob_iters 200 --plot_legend 1 --test_acc True





  




