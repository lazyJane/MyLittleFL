 python main_plot.py --dataset Mnist-alpha0.4-ratio0.5 --algorithms FedIFCA,FedAvg,FedKLEM,Fedlocal,FedProx,FedKLtopk\
  --batch_size 32 \
 --local_epochs 20 --num_users 20 --num_glob_iters 200 --plot_legend 1 --test_acc True


  python main_plot.py --dataset Mnist-pathological-ratio0.2 --algorithms FedIFCA,FedAvg,FedKLEM,Fedlocal,FedProx,FedKLtopk\
  --batch_size 32 \
 --local_epochs 20 --num_users 20 --num_glob_iters 200 --plot_legend 1 --E 1 --test_acc True


  python main_plot.py --dataset Cifar10-pathological-ratio0.1 --algorithms \
 Fedlocal,FedProx,FedKLtopk,FedIFCA,FedAvg,FedKLEM --batch_size 32  \
  --num_users 30 --num_glob_iters 200 --plot_legend 1 --E 1 --test_acc True
 

  python main_plot.py --dataset  Cifar10-label_swapped-ratio0.1 --algorithms \
  Fedlocal,FedKLtopk,FedIFCA,FedAvg,FedKLEM --batch_size 32  \
  --num_users 20 --num_glob_iters 200 --plot_legend 1 --E 1 --test_acc True



 python main_plot.py --dataset Cifar10-alpha0.4-ratio0.1 --algorithms \
 Fedlocal,FedKLtopk,FedIFCA,FedAvg,FedKLEM --batch_size 32  --local_epochs 20 \
  --num_users 30 --num_glob_iters 200 --plot_legend 1 --E 1 --test_acc True

 python main_plot.py --dataset EMnist-alpha0.4-ratio0.02 --algorithms FedAvg,FedKLEM,FedIFCA\
 --batch_size 32 --E 1 --num_users 50 --num_glob_iters 200 --plot_legend 1 --test_acc True

  python main_plot.py --dataset FEMNIST --algorithms PerFedavg,FedAvg,FedKLEM,FedIFCA,FedKLtopk,FedProx,FedSEM,Fedlocal\
 --batch_size 32 --local_epochs 20 --num_users 35 --num_glob_iters 500 --plot_legend 1 --test_acc True

   python main_plot.py --dataset FEMNIST --algorithms PerFedavg,FedAvg,FedKLEM,FedIFCA,FedKLtopk,FedProx,FedSEM,Fedlocal\
 --batch_size 32 --local_epochs 20 --num_users 35 --num_glob_iters 500 --plot_legend 1 --test_acc True

    python main_plot.py --dataset FEMNIST --algorithms FedAvg,FedKLEM,FedIFCA,FedKLtopk,FedProx\
 --batch_size 32 --E 3 --num_users 35 --num_glob_iters 500 --plot_legend 1 --test_acc True