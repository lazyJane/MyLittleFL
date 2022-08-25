python main_plot.py --dataset Mnist-alpha0.5-ratio1.0-u100 --algorithms FedKLEM,FedAvg,FedIFCA,FedKLEM_PerFedAvg_ratio,FedKLEM_PerFedAvg_ratio_e,PerFedavg,FedSEM \
--batch_size 32 --local_epochs 10 --num_users 80 --num_glob_iters 200 --plot_legend 1 --unseen_test_acc True 
