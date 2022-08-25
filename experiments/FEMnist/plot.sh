
 python main_plot.py --dataset FEMNIST --algorithms FedKLEM,FedIFCA,FedSEM,PerFedavg,FedKLEM_PerFedAvg_ratio_e \
  --batch_size 32 --E 3 --num_users 287 --num_glob_iters 200 --plot_legend 1 --test_acc True 


 python main_plot.py --dataset FEMNIST --algorithms  Fedlocal,FedKLEM_PerFedAvg,FedAvg,FedProx,FedIFCA,FedSEM,PerFedavg,FedKLEM,FedKLEM_PerFedAvg_ratio,FedKLEM_PerFedAvg_ratio_e \
  --batch_size 32 --E 3 --num_users 143 --num_glob_iters 200 --plot_legend 1 --test_acc True 

 python main_plot.py --dataset FEMNIST --algorithms  FedAvg,FedProx,FedIFCA,FedKLEM,FedKLEM_PerFedAvg,FedKLEM_PerFedAvg_ratio,FedKLEM_PerFedAvg_ratio_e\
  --batch_size 32 --E 3 --num_users 58 --num_glob_iters 200 --plot_legend 1 --test_acc True 


   python main_plot.py --dataset FEMNIST --algorithms FedAvg,FedProx,FedIFCA,FedSEM,PerFedavg,FedKLEM,FedKLEM_PerFedAvg,FedKLEM_PerFedAvg_ratio,FedKLEM_PerFedAvg_ratio_e \
  --batch_size 32 --E 3 --num_users 70 --num_glob_iters 200 --plot_legend 1 --test_acc True

     python main_plot.py --dataset FEMNIST --algorithms Fedlocal,FedAvg,FedProx,FedKLEM_PerFedAvg_ratio,FedIFCA,FedSEM,FedKLEM,PerFedavg,FedKLEM_PerFedAvg_ratio_e,FedKLEM_PerFedAvg\
  --batch_size 32 --E 3 --num_users 35 --num_glob_iters 200 --plot_legend 1 --bottom_acc True 

     python main_plot.py --dataset FEMNIST --algorithms FedAvg,FedProx,FedKLEM_PerFedAvg_ratio,FedIFCA,FedSEM,FedKLEM,PerFedavg,FedKLEM_PerFedAvg_ratio_e,FedKLEM_PerFedAvg\
  --batch_size 32 --E 3 --num_users 14 --num_glob_iters 200 --plot_legend 1 --bottom_acc True


   python main_plot.py --dataset FEMNIST --algorithms FedAvg,FedProx,FedIFCA,FedSEM,PerFedavg,FedKLEM,FedKLEM_PerFedAvg,FedKLEM_PerFedAvg_ratio,FedKLEM_PerFedAvg_ratio_e \
  --batch_size 32 --E 3 --num_users 70 --num_glob_iters 200 --plot_legend 1 --test_acc True