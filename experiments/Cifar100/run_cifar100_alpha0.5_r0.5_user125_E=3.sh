 nohup python -u main.py --dataset Cifar100-alpha0.5-ratio0.5-u125 --algorithm FedAvg --batch_size 32 \
 --num_users 100 --learning_rate 0.01 --num_glob_iters 25 --E 3 --times 3 --gpu 0 > ./acc_loss_record/cifar100_user125_unseen25/E=3/FedAvg.out 2>&1 &

 nohup python -u main.py --dataset Cifar100-alpha0.5-ratio0.5-u125 --algorithm FedKLEM --batch_size 32 \
 --num_users 100 --learning_rate 0.01 --num_glob_iters 25 --E 3 --times 3 --gpu 1 > ./acc_loss_record/cifar100_user125_unseen25/E=3/FedKLEM.out 2>&1 &

 nohup python -u main.py --dataset Cifar100-alpha0.5-ratio0.5-u125 --algorithm FedIFCA --batch_size 32 \
 --num_users 100 --learning_rate 0.01 --num_glob_iters 25 --E 3 --times 3 --gpu 2 > ./acc_loss_record/cifar100_user125_unseen25/E=3/FedIFCA.out 2>&1 &

 nohup python -u main.py --dataset Cifar100-alpha0.5-ratio0.5-u125 --algorithm FedKLEM_PerFedAvg --batch_size 32 \
 --num_users 100 --beta 0.01 --learning_rate 0.01 --num_glob_iters 25 --E 3 --times 3 --gpu 3 > ./acc_loss_record/cifar100_user125_unseen25/E=3/FedKLEM_PerFedAvg.out 2>&1 &

 nohup python -u main.py --dataset Cifar100-alpha0.5-ratio0.5-u125 --algorithm FedKLEM_PerFedAvg_ratio --batch_size 32 \
 --num_users 100 --beta 0.01 --learning_rate 0.01 --num_glob_iters 25 --E 3 --times 3 --gpu 4 > ./acc_loss_record/cifar100_user125_unseen25/E=3/FedKLEM_PerFedAvg_ratio.out 2>&1 &

 nohup python -u main.py --dataset Cifar100-alpha0.5-ratio0.5-u125 --algorithm FedKLEM_PerFedAvg_ratio_e --batch_size 32 \
 --num_users 100 --beta 0.01 --learning_rate 0.01 --num_glob_iters 25 --E 3 --times 3 --gpu 5 > ./acc_loss_record/cifar100_user125_unseen25/E=3/FedKLEM_PerFedAvg_ratio_e.out 2>&1 &

 nohup python -u main.py --dataset Cifar100-alpha0.5-ratio0.5-u125 --algorithm FedSEM --batch_size 32 \
 --num_users 100 --learning_rate 0.01 --num_glob_iters 25 --E 3 --times 3 --gpu 2 > ./acc_loss_record/cifar100_user125_unseen25/E=3/FedSEM.out 2>&1 &

 nohup python -u main.py --dataset Cifar100-alpha0.5-ratio0.5-u125 --algorithm PerFedavg --batch_size 32 \
 --num_users 100 --beta 0.01 --learning_rate 0.01 --num_glob_iters 25 --E 3 --times 3 --gpu 3 > ./acc_loss_record/cifar100_user125_unseen25/E=3/PerFedavg.out 2>&1 &

  