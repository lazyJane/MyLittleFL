  python main_plot.py --dataset Mnist-alpha0.5-ratio1.0-u100 --algorithms Fedlocal,FedProx,Fedlocal,FedAvg,FedIFCA,FedKLEM,FedKLEM_PerFedAvg_ratio,FedKLEM_PerFedAvg_ratio_e,PerFedavg,FedSEM\
  --batch_size 32 --local_epochs 20 --num_users 80 --num_glob_iters 200 --plot_legend 1 --bottom_acc True 

    python main_plot.py --dataset Mnist-alpha0.5-ratio1.0-u100 --algorithms FedKLEM_PerFedAvg_ratio,FedKLEM_PerFedAvg_ratio_e,PerFedavg,FedSEM\
  --batch_size 32 --local_epochs 20 --num_users 80 --num_glob_iters 200 --plot_legend 1 --bottom_acc True 

      python main_plot.py --dataset Mnist-alpha0.5-ratio1.0-u100 --algorithms FedKLEM_PerFedAvg_ratio,FedKLEM_PerFedAvg_ratio_e,PerFedavg,FedSEM\
  --batch_size 32 --local_epochs 40 --num_users 80 --num_glob_iters 200 --plot_legend 1 --bottom_acc True 

        python main_plot.py --dataset Mnist-alpha0.5-ratio1.0-u100 --algorithms FedKLEM_PerFedAvg_ratio,FedKLEM_PerFedAvg_ratio_e,PerFedavg,FedSEM\
  --batch_size 32 --E 1 --num_users 80 --num_glob_iters 200 --plot_legend 1 --bottom_acc True 

          python main_plot.py --dataset Mnist-alpha0.5-ratio1.0-u100 --algorithms FedKLEM_PerFedAvg_ratio,FedKLEM_PerFedAvg_ratio_e,PerFedavg,FedSEM\
  --batch_size 32 --E 3 --num_users 80 --num_glob_iters 200 --plot_legend 1 --test_acc True 

    python main_plot.py --dataset Mnist-alpha1.0-ratio1.0-u100 --algorithms FedKLEM_PerFedAvg,FedSEM,Fedlocal,FedProx,FedKLEM,FedAvg,FedIFCA,FedKLEM_PerFedAvg_ratio,FedKLEM_PerFedAvg_ratio_e,PerFedavg\
  --batch_size 32 --local_epochs 10 --num_users 80 --num_glob_iters 200 --plot_legend 1 --test_acc True --times 3

      python main_plot.py --dataset Mnist-alpha10.0-ratio1.0-u100 --algorithms FedKLEM,FedKLEM_PerFedAvg,FedSEM,Fedlocal,FedProx,FedKLEM,FedAvg,FedIFCA,FedKLEM_PerFedAvg_ratio,FedKLEM_PerFedAvg_ratio_e,PerFedavg\
  --batch_size 32 --local_epochs 10 --num_users 80 --num_glob_iters 200 --plot_legend 1 --bottom_acc True --times 3

        python main_plot.py --dataset Mnist-alpha10.0-ratio1.0-u100 --algorithms FedKLEM_PerFedAvg,FedSEM,FedKLEM,FedIFCA,FedKLEM_PerFedAvg_ratio,FedKLEM_PerFedAvg_ratio_e,PerFedavg\
  --batch_size 32 --E 3 --num_users 80 --num_glob_iters 200 --plot_legend 1 --test_acc True --times 1

    python main_plot.py --dataset Mnist-alpha5.0-ratio1.0-u100 --algorithms Fedlocal,FedProx,Fedlocal,FedAvg,FedIFCA,FedKLEM,FedKLEM_PerFedAvg_ratio,FedKLEM_PerFedAvg_ratio_e,PerFedavg,FedSEM\
  --batch_size 32 --local_epochs 10 --num_users 80 --num_glob_iters 200 --plot_legend 1 --bottom_acc_1 True 

      python main_plot.py --dataset Mnist-alpha0.5-ratio1.0-u100 --algorithms FedKLEM_PerFedAvg_ratio_e \
  --batch_size 32 --local_epochs 10 --num_users 80 --num_glob_iters 200 --plot_legend 1 --test_acc True --times 3

  