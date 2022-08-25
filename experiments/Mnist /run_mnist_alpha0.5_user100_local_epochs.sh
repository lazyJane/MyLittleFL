  python main.py --dataset Mnist-alpha0.5-ratio1.0-u100 --algorithm FedAvg \
  --batch_size 32 --num_users 100 --learning_rate 0.01 --num_glob_iters 500 --local_epochs 10 --times 1

  python main.py --dataset Mnist-alpha0.5-ratio1.0-u100 --algorithm FedKLEM \
  --batch_size 32 --num_users 100 --learning_rate 0.01 --num_glob_iters 500 --local_epochs 10 --times 1 

    python main.py --dataset Mnist-alpha0.5-ratio1.0-u100 --algorithm FedKLEM_IntraC \
  --batch_size 32 --beta 0.01 --num_users 100 --learning_rate 0.01 --num_glob_iters 500 --local_epochs 10 --times 1 


    python main.py --dataset Mnist-alpha0.5-ratio1.0-u100 --algorithm FedKLEM_intra_cluster_PerFedavg \
  --batch_size 32 --beta 0.01 --num_users 100 --learning_rate 0.01 --num_glob_iters 500 --local_epochs 10 --times 1 

  python main.py --dataset Mnist-alpha0.5-ratio1.0-u100 --algorithm FedSEM \
  --batch_size 32 --num_users 100 --learning_rate 0.01 --num_glob_iters 500 --local_epochs 10 --times 1 

  python main.py --dataset Mnist-alpha0.5-ratio1.0-u100 --algorithm FedIFCA \
  --batch_size 32 --num_users 100 --learning_rate 0.01 --num_glob_iters 500 --local_epochs 10 --times 1 

    python main.py --dataset Mnist-alpha0.5-ratio1.0-u100 --algorithm FedKLEM_PerFedAvg\
  --batch_size 32 --beta 0.01 --num_users 100 --learning_rate 0.01 --num_glob_iters 500 --local_epochs 10 --times 1

      python main.py --dataset Mnist-alpha0.5-ratio1.0-u100 --algorithm PerFedavg\
  --batch_size 32 --beta 0.01 --num_users 100 --learning_rate 0.01 --num_glob_iters 500 --local_epochs 10 --times 1

  python main.py --dataset Mnist-alpha0.5-ratio1.0-u100 --algorithm FedCluster \
  --batch_size 32 --num_users 100 --learning_rate 0.01 --num_glob_iters 200 --E 1 --times 1

 python main.py --dataset Mnist-alpha0.5-ratio1.0-u100 --algorithm FedProx \
 --batch_size 32 --num_users 10 --learning_rate 0.01 --num_glob_iters 500 --E 1 --times 1

 python main.py --dataset Mnist-alpha0.5-ratio1.0-u100 --algorithm Fedlocal \
 --batch_size 32 --num_users 100 --learning_rate 0.01 --num_glob_iters 500 --local_epochs 10  --times 1

 python main.py --dataset Mnist-alpha0.5-ratio1.0-u100 --algorithm FedKLtopk \
 --batch_size 32 --num_users 100 --learning_rate 0.01 --num_glob_iters 500 --E 1 --times 1

 python main.py --dataset Mnist-alpha0.5-ratio1.0-u100 --algorithm pFedME \
 --batch_size 32 --num_users 100 --learning_rate 0.01 --num_glob_iters 500 --local_epochs 10 \
 --personal_learning_rate 0.01 --beta 1 --lamda 15 --times 1


  python main_plot.py --dataset Mnist-alpha0.5-ratio1.0-u100 --algorithms FedKLtopk,FedKLEM_PerFedAvg,PerFedavg,FedIFCA,FedKLEM,FedAvg\
  --batch_size 32 --local_epochs 10 --num_users 80 --num_glob_iters 200 --plot_legend 1 --test_acc True

      python main_plot.py --dataset Mnist-alpha0.5-ratio1.0-u100 --algorithms FedKLEM_intra_cluster_PerFedavg,FedKLEM_PerFedAvg,PerFedavg,FedKLEM,FedAvg\
  --batch_size 32 --local_epochs 10 --num_users 100 --num_glob_iters 200 --plot_legend 1 --test_acc True

      python main_plot.py --dataset Mnist-alpha0.5-ratio1.0-u100 --algorithms FedKLEM_IntraC,FedKLEM\
  --batch_size 32 --local_epochs 10 --num_users 100 --num_glob_iters 200 --plot_legend 1 --test_acc True

      python main_plot.py --dataset Mnist-alpha0.5-ratio1.0-u100 --algorithms PerFedavg,FedIFCA,FedKLEM,FedAvg\
  --batch_size 32 --E 3 --num_users 100 --num_glob_iters 200 --plot_legend 1 --test_acc True

      python main_plot.py --dataset Mnist-alpha0.5-ratio1.0-u100 --algorithms FedKLEM_IntraCpFedME,FedProx,FedKLEM_PerFedAvg,PerFedavg,FedIFCA,FedKLEM,FedAvg,FedSEM\
  --batch_size 32 --E 1 --num_users 10 --num_glob_iters 200 --plot_legend 1 --test_acc True