
 nohup python -u main.py --dataset Cifar100-alpha0.5-ratio1.0-u100 --algorithm FedAvg --batch_size 32 \
 --num_users 80 --learning_rate 0.01 --num_glob_iters 300 --local_epochs 10  --times 1 > FedAvg.out 2>&1 &
 
 nohup python -u main.py --dataset Cifar100-alpha0.5-ratio1.0-u100 --algorithm FedKLEM --batch_size 32 \
 --num_users 80 --learning_rate 0.01 --num_glob_iters 300 --local_epochs 10  --times 1 --gpu 1 > FedKLEM.out 2>&1 &

  nohup python main.py --dataset Cifar100-alpha0.5-ratio1.0-u100 --algorithm FedKLEM_IntraC --batch_size 32 \
 --beta 0.01 --num_users 80 --learning_rate 0.01 --num_glob_iters 300 --E 1  --times 1 --p 3

  nohup python main.py --dataset Cifar100-alpha0.5-ratio1.0-u100 --algorithm  FedKLEM_intra_cluster_PerFedavg --batch_size 32 \
 --beta 0.01 --num_users 80 --learning_rate 0.01 --num_glob_iters 300 --E 1  --times 1 --p 3

 nohup python -u main.py --dataset Cifar100-alpha0.5-ratio1.0-u100 --algorithm FedSEM --batch_size 32 \
 --num_users 80 --learning_rate 0.01 --num_glob_iters 300 --local_epochs 10 --times 1 --gpu 5 > FedSEM.out 2>&1 &

  nohup python -u main.py --dataset Cifar100-alpha0.5-ratio1.0-u100 --algorithm FedIFCA --batch_size 32 \
--num_users 80 --learning_rate 0.01 --num_glob_iters 300 --local_epochs 10  --times 1 --gpu 4 > FedIFCA.out 2>&1 &

   nohup python -u main.py --dataset Cifar100-alpha0.5-ratio1.0-u100 --algorithm FedKLEM_PerFedAvg --batch_size 32 \
 --num_users 80 --beta 0.01 --learning_rate 0.01 --num_glob_iters 300 --local_epochs 10 --times 1 --gpu 2 > FedKLEM_PerFedAvg.out 2>&1 &

  nohup   python main.py --dataset Cifar100-alpha0.5-ratio1.0-u100 --algorithm Fedlocal --batch_size 32 \
 --num_users 80 --learning_rate 0.01 --num_glob_iters 300 --local_epochs 10  --times 1

  nohup    python main.py --dataset Cifar100-alpha0.5-ratio1.0-u100 --algorithm FedProx --batch_size 32 \
 --num_users 80 --learning_rate 0.01 --num_glob_iters 300 --local_epochs 10  --times 1

  nohup python -u main.py --dataset  Cifar100-alpha0.5-ratio1.0-u100 --algorithm PerFedavg \
  --batch_size 32 --num_users 80 --beta 0.01 --learning_rate 0.01 --num_glob_iters 300 --local_epochs 10  --times 1 --gpu 3 > PerFedAvg.out 2>&1 &

  nohup python main.py --dataset  Cifar100-alpha0.5-ratio1.0-u100 --algorithm pFedME \
 --batch_size 32 --num_users 80 --learning_rate 0.01 --num_glob_iters 300 --E 1  \
 --personal_learning_rate 0.01 --beta 1 --lamda 15 --times 1


 python main_plot.py --dataset  Cifar100-alpha0.5-ratio1.0-u100 --algorithms FedSEM,PerFedavg,FedKLEM_PerFedAvg,FedAvg,FedIFCA,FedKLEM\
  --batch_size 32 --local_epochs 10  --num_users 80 --num_glob_iters 300 --plot_legend 1 --test_acc True

  python main_plot.py --dataset  Cifar100-alpha0.5-ratio1.0-u100 --algorithms Fedlocal\
  --batch_size 32 --E 1  --num_users 80 --num_glob_iters 200 --plot_legend 1 --test_acc True

   python main_plot.py --dataset  Cifar100-alpha0.5-ratio1.0-u100 --algorithms PerFedavg\
  --batch_size 32 --E 1   --num_users 80 --num_glob_iters 200 --plot_legend 1 --test_acc True

   nohup python main.py --dataset Cifar100-alpha0.5-ratio1.0-u100 --algorithm FedfuzzyC --batch_size 32 \
 --num_users 80 --learning_rate 0.01 --num_glob_iters 300 --E 1  --times 1

   nohup  python main.py --dataset Cifar100-alpha0.5-ratio1.0-u100 --algorithm FedKLtopk --batch_size 32 \
 --num_users 80 --learning_rate 0.01 --num_glob_iters 300 --E 1 --times 1
 