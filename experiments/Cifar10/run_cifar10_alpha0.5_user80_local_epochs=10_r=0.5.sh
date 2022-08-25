 nohup python -u main.py --dataset Cifar10-alpha0.5-ratio0.5-u80 --algorithm FedAvg --batch_size 32 \
 --num_users 32 --learning_rate 0.01 --num_glob_iters 200 --local_epochs 10 --times 1 --gpu 1 > ./acc_loss_record/cifar10/local_epochs=10/r=0.5/FedAvg.out 2>&1 &

 nohup python -u main.py --dataset Cifar10-alpha0.5-ratio0.5-u80 --algorithm FedKLEM --batch_size 32 \
 --num_users 32 --learning_rate 0.01 --num_glob_iters 200 --local_epochs 10 --times 1 --gpu 2 > ./acc_loss_record/cifar10/local_epochs=10/r=0.5/FedKLEM.out 2>&1 &

 nohup python -u main.py --dataset Cifar10-alpha0.5-ratio0.5-u80 --algorithm FedIFCA --batch_size 32 \
 --num_users 32 --learning_rate 0.01 --num_glob_iters 200 --local_epochs 10 --times 1 --gpu 2 > ./acc_loss_record/cifar10/local_epochs=10/r=0.5/FedIFCA.out 2>&1 &

 nohup python -u main.py --dataset Cifar10-alpha0.5-ratio0.5-u80 --algorithm FedKLEM_PerFedAvg --batch_size 32 \
 --num_users 32 --beta 0.01 --learning_rate 0.01 --num_glob_iters 200 --local_epochs 10 --times 1 --gpu 2 > ./acc_loss_record/cifar10/local_epochs=10/r=0.5/FedKLEM_PerFedAvg.out 2>&1 &

 nohup python -u main.py --dataset Cifar10-alpha0.5-ratio0.5-u80 --algorithm FedSEM --batch_size 32 \
 --num_users 32 --learning_rate 0.01 --num_glob_iters 200 --local_epochs 10 --times 1 --gpu 4 > ./acc_loss_record/cifar10/local_epochs=10/r=0.5/FedSEM.out 2>&1 &

 nohup python -u main.py --dataset Cifar10-alpha0.5-ratio0.5-u80 --algorithm PerFedavg --batch_size 32 \
 --num_users 32 --beta 0.01 --learning_rate 0.01 --num_glob_iters 200 --local_epochs 10 --times 1 --gpu 5 > ./acc_loss_record/cifar10/local_epochs=10/r=0.5/PerFedAvg.out 2>&1 &


  