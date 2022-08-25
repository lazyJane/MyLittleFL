
 python main.py --dataset Cifar100-alpha0.5-ratio0.5-u80 --algorithm FedAvg --batch_size 32 \
 --num_users 80 --learning_rate 0.01 --num_glob_iters 200 --E 1  --times 1
 
 python main.py --dataset Cifar100-alpha0.5-ratio0.5-u80 --algorithm FedKLEM --batch_size 32 \
 --num_users 80 --learning_rate 0.01 --num_glob_iters 200 --E 1  --times 1

 python main.py --dataset Cifar100-alpha0.5-ratio0.5-u80 --algorithm FedfuzzyC --batch_size 32 \
 --num_users 80 --learning_rate 0.01 --num_glob_iters 200 --E 1  --times 1


  python main.py --dataset Cifar100-alpha0.5-ratio0.5-u80 --algorithm FedKLEM_IntraC --batch_size 32 \
 --beta 0.01 --num_users 80 --learning_rate 0.01 --num_glob_iters 200 --E 1  --times 1 --p 3


  python main.py --dataset Cifar100-alpha0.5-ratio0.5-u80 --algorithm  FedKLEM_intra_cluster_PerFedavg --batch_size 32 \
 --beta 0.01 --num_users 80 --learning_rate 0.01 --num_glob_iters 200 --E 1  --times 1 --p 3

 python main.py --dataset Cifar100-alpha0.5-ratio0.5-u80 --algorithm FedSEM --batch_size 32 \
 --num_users 80 --learning_rate 0.01 --num_glob_iters 200 --E 1 --times 1

  python main.py --dataset Cifar100-alpha0.5-ratio0.5-u80 --algorithm FedIFCA --batch_size 32 \
--num_users 80 --learning_rate 0.01 --num_glob_iters 200 --E 1  --times 1

    python main.py --dataset Cifar100-alpha0.5-ratio0.5-u80 --algorithm FedKLEM_PerFedAvg --batch_size 32 \
 --num_users 80 --beta 0.01 --learning_rate 0.01 --num_glob_iters 200 --E 1 --times 1

   python main.py --dataset Cifar100-alpha0.5-ratio1.0-u80 --algorithm FedKLtopk --batch_size 32 \
 --num_users 80 --learning_rate 0.01 --num_glob_iters 200 --E 1 --times 1

    python main.py --dataset Cifar100-alpha0.5-ratio0.5-u80 --algorithm Fedlocal --batch_size 32 \
 --num_users 80 --learning_rate 0.01 --num_glob_iters 200 --E 1  --times 1

     python main.py --dataset Cifar100-alpha0.5-ratio0.5-u80 --algorithm FedProx --batch_size 32 \
 --num_users 80 --learning_rate 0.01 --num_glob_iters 200 --E 1  --times 1

 python main.py --dataset  Cifar100-alpha0.5-ratio0.5-u80 --algorithm PerFedavg \
  --batch_size 32 --num_users 80 --beta 0.01 --learning_rate 0.01 --num_glob_iters 200 --E 1  --times 1

  python main.py --dataset  Cifar100-alpha0.5-ratio0.5-u80 --algorithm pFedME \
 --batch_size 32 --num_users 80 --learning_rate 0.01 --num_glob_iters 200 --E 1  \
 --personal_learning_rate 0.01 --beta 1 --lamda 15 --times 1


 python main_plot.py --dataset  Cifar100-alpha0.5-ratio0.5-u80 --algorithms Fedlocal,FedSEM,FedProx,FedKLEM_PerFedAvg,FedKLEM_intra_cluster_PerFedavg,PerFedavg,FedAvg,FedIFCA,FedKLEM\
  --batch_size 32 --E 1  --num_users 80 --num_glob_iters 200 --plot_legend 1 --test_acc True

  python main_plot.py --dataset  Cifar100-alpha0.5-ratio1.0-u80 --algorithms Fedlocal\
  --batch_size 32 --E 1  --num_users 80 --num_glob_iters 200 --plot_legend 1 --test_acc True

   python main_plot.py --dataset  Cifar100-alpha0.5-ratio0.5-u80 --algorithms PerFedavg\
  --batch_size 32 --E 1   --num_users 80 --num_glob_iters 200 --plot_legend 1 --test_acc True
 