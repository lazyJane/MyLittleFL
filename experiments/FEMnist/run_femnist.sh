python main.py --dataset FEMNIST --algorithm FedAvg --batch_size 32 \
--num_users 70 --learning_rate 0.01 --num_glob_iters 500 --E 1 --times 1 

python main.py --dataset FEMNIST --algorithm FedKLEM --batch_size 32 \
--num_users 70 --learning_rate 0.01 --num_glob_iters 500 --E 5 --times 1 --p 3

 python main.py --dataset FEMNIST --algorithm FedKLEM_PerFedAvg --batch_size 32 \
 --num_users 70 --beta 0.01 --learning_rate 0.01 --num_glob_iters 500 --E 3 --times 1 --p 3

   python main.py --dataset FEMNIST --algorithm FedKLEM_IntraC --batch_size 32 \
 --beta 0.001 --num_users 70 --learning_rate 0.01 --num_glob_iters 200 --E 3  --times 1 --p 3

 python main.py --dataset FEMNIST --algorithm FedKLEM_intra_cluster_PerFedavg\
  --batch_size 32 --beta 0.001 --num_users 70 --learning_rate 0.01 --num_glob_iters 500 --E 3 --times 1

python main.py --dataset FEMNIST --algorithm FedfuzzyC --batch_size 32 \
--num_users 70 --learning_rate 0.01 --num_glob_iters 500 --E 3 --times 1 --p 3

python main.py --dataset FEMNIST --algorithm FedIFCA --batch_size 32 \
--num_users 70 --learning_rate 0.01 --num_glob_iters 500 --E 3 --times 1 --p 3

python main.py --dataset FEMNIST --algorithm FedSEM --batch_size 32 \
--num_users 70 --learning_rate 0.01 --num_glob_iters 500 --E 3 --times 1 --p 3

python main.py --dataset FEMNIST  --algorithm FedProx --batch_size 32 \
--num_users 70 --learning_rate 0.01 --num_glob_iters 500 --E 3 --times 1 

python main.py --dataset FEMNIST --algorithm Fedlocal --batch_size 32 \
--num_users 70 --learning_rate 0.01 --num_glob_iters 500 --E 3 --times 1 

python main.py --dataset FEMNIST --algorithm FedKLtopk --batch_size 32 \
--num_users 70 --learning_rate 0.01 --num_glob_iters 500 --E 3 --times 1 

python main.py --dataset FEMNIST --algorithm FedSEM --batch_size 32 \
--num_users 358 --learning_rate 0.01 --num_glob_iters 500 --E 3 --times 1

python main.py --dataset FEMNIST --algorithm PerFedavg --batch_size 32 \
 --num_users 70 --beta 0.01 --learning_rate 0.01 --num_glob_iters 500 --E 3 --times 1

python main.py --dataset FEMNIST --algorithm pFedME --batch_size 32 \
 --num_users 70 --beta 2 --lamda 15 --learning_rate 0.01 \
--personal_learning_rate 0.01 --num_glob_iters 500 --E 3 --times 1


 python main_plot.py --dataset FEMNIST --algorithms  FedKLEM_intra_cluster_PerFedavg,FedKLEM_IntraC,FedProx,PerFedavg,FedKLEM_PerFedAvg,FedSEM,FedAvg,FedIFCA,FedKLEM\
  --batch_size 32 --E 3 --num_users 70 --num_glob_iters 200 --plot_legend 1 --test_acc True

  python main_plot.py --dataset FEMNIST --algorithms  FedKLEM_intra_cluster_PerFedavg,PerFedavg,FedKLEM_PerFedAvg,FedSEM,FedAvg,FedIFCA,FedKLEM\
  --batch_size 32 --E 3 --num_users 70 --num_glob_iters 200 --plot_legend 1 --test_acc True
  
 python main_plot.py --dataset FEMNIST --algorithms  FedSEM,FedAvg,FedIFCA,FedKLEM\
  --batch_size 32 --E 1 --num_users 70 --num_glob_iters 200 --plot_legend 1 --test_acc True
