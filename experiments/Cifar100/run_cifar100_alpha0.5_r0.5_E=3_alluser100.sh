 nohup python -u main.py --dataset Cifar100-alpha0.5-ratio0.5-u100 --algorithm FedAvg --batch_size 32 \
 --num_users 80 --learning_rate 0.01 --num_glob_iters 200 --E 3 --times 1 --gpu 0 > ./acc_loss_record/cifar100_r1.0/E=3/FedAvg.out 2>&1 &

 nohup python -u main.py --dataset Cifar100-alpha0.5-ratio0.5-u100 --algorithm FedKLEM --batch_size 32 \
 --num_users 80 --learning_rate 0.01 --num_glob_iters 200 --E 3 --times 1 --gpu 1 > ./acc_loss_record/cifar100_r0.5/E=3/FedKLEM.out 2>&1 &

 nohup python -u main.py --dataset Cifar100-alpha0.5-ratio0.5-u100 --algorithm FedIFCA --batch_size 32 \
 --num_users 80 --learning_rate 0.01 --num_glob_iters 200 --E 3 --times 1 --gpu 4 > ./acc_loss_record/cifar100_r0.5/E=3/FedIFCA.out 2>&1 &

 nohup python -u main.py --dataset Cifar100-alpha0.5-ratio0.5-u100 --algorithm FedKLEM_PerFedAvg --batch_size 32 \
 --num_users 80 --beta 0.01 --learning_rate 0.01 --num_glob_iters 200 --E 3 --times 1 --gpu 2 > ./acc_loss_record/cifar100_r0.5/E=3/FedKLEM_PerFedAvg.out 2>&1 &

 nohup python -u main.py --dataset Cifar100-alpha0.5-ratio0.5-u100 --algorithm FedSEM --batch_size 32 \
 --num_users 80 --learning_rate 0.01 --num_glob_iters 200 --E 3 --times 1 --gpu 4 > ./acc_loss_record/cifar100_r0.5/E=3/FedSEM.out 2>&1 &

 nohup python -u main.py --dataset Cifar100-alpha0.5-ratio0.5-u100 --algorithm PerFedavg --batch_size 32 \
 --num_users 80 --beta 0.01 --learning_rate 0.01 --num_glob_iters 200 --E 3 --times 1 --gpu 5 > ./acc_loss_record/cifar100_r0.5/E=3/PerFedAvg.out 2>&1 &


  