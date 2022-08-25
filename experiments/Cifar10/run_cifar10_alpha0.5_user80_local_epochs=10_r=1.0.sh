 nohup python -u main.py --dataset Cifar10-alpha0.5-ratio0.5-u80 --algorithm FedAvg --batch_size 32 \
 --num_users 6 --learning_rate 0.01 --num_glob_iters 200 --local_epochs 10 --times 1 --gpu 1 > ./acc_loss_record/cifar10/local_epochs=10/r=0.1/FedAvg.out 2>&1 &

 nohup python -u main.py --dataset Cifar10-alpha0.5-ratio0.5-u80 --algorithm FedKLEM --batch_size 32 \
 --num_users 80 --learning_rate 0.01 --num_glob_iters 200 --local_epochs 10 --times 1 --gpu 3 > ./acc_loss_record/cifar10/local_epochs=20/FedKLEM.out 2>&1 &

 nohup python -u main.py --dataset Cifar10-alpha0.5-ratio0.5-u80 --algorithm FedIFCA --batch_size 32 \
 --num_users 80 --learning_rate 0.01 --num_glob_iters 200 --local_epochs 10 --times 1 --gpu 2 > ./acc_loss_record/cifar10/local_epochs=20/FedIFCA.out 2>&1 &

 nohup python -u main.py --dataset Cifar10-alpha0.5-ratio0.5-u80 --algorithm FedKLEM_PerFedAvg --batch_size 32 \
 --num_users 6 --beta 0.01 --learning_rate 0.01 --num_glob_iters 200 --local_epochs 10 --times 1 --gpu 3 > ./acc_loss_record/cifar10/local_epochs=10/r=0.1/FedKLEM_PerFedAvg.out 2>&1 &

 nohup python -u main.py --dataset Cifar10-alpha0.5-ratio0.5-u80 --algorithm FedSEM --batch_size 32 \
 --num_users 80 --learning_rate 0.01 --num_glob_iters 200 --local_epochs 10 --times 1 --gpu 3 > ./acc_loss_record/cifar10/local_epochs=10/FedSEM.out 2>&1 &

 nohup python -u main.py --dataset Cifar10-alpha0.5-ratio0.5-u80 --algorithm PerFedavg --batch_size 32 \
 --num_users 80 --beta 0.01 --learning_rate 0.01 --num_glob_iters 200 --local_epochs 10 --times 3 --gpu 5 > ./acc_loss_record/cifar10/local_epochs=10/PerFedAvg.out 2>&1 &


  nohup python -u main.py --dataset Cifar10-alpha0.5-ratio0.5-u80 --algorithm FedKLEM_PerFedAvg_ratio --batch_size 32 \
 --num_users 80 --beta 0.01 --learning_rate 0.01 --num_glob_iters 200 --local_epochs 10 --times 3 --gpu 2 > ./acc_loss_record/cifar10/local_epochs=10/FedKLEM_PerFedAvg_ratio.out 2>&1 &

   nohup python -u main.py --dataset Cifar10-alpha0.5-ratio0.5-u80 --algorithm FedKLEM_PerFedAvg_ratio_e --batch_size 32 \
 --num_users 80 --beta 0.01 --learning_rate 0.01 --num_glob_iters 200 --local_epochs 10 --times 3 --gpu 5 > ./acc_loss_record/cifar10/local_epochs=10/FedKLEM_PerFedAvg_ratio_e.out 2>&1 &

   nohup python -u main.py --dataset Cifar10-alpha0.5-ratio0.5-u80 --algorithm FedKLEM_PerFedAvg --batch_size 32 \
 --num_users 80 --beta 0.01 --learning_rate 0.01 --num_glob_iters 200 --local_epochs 10 --times 3 --gpu 3 > ./acc_loss_record/cifar10/local_epochs=10/FedKLEM_PerFedAvg.out 2>&1 &

 nohup python -u main.py --dataset Cifar10-alpha0.5-ratio0.5-u80 --algorithm FedKLEM_PerFedAvg_ratio --batch_size 32 \
 --num_users 80 --beta 0.01 --learning_rate 0.01 --num_glob_iters 200 --local_epochs 20 --times 3 --gpu 2 > ./acc_loss_record/cifar10/local_epochs=20/FedKLEM_PerFedAvg_ratio.out 2>&1 &

    nohup python -u main.py --dataset Cifar10-alpha0.5-ratio0.5-u80 --algorithm FedKLEM_PerFedAvg_ratio_e --batch_size 32 \
 --num_users 80 --beta 0.01 --learning_rate 0.01 --num_glob_iters 200 --local_epochs 20 --times 3 --gpu 3 > ./acc_loss_record/cifar10/local_epochs=20/FedKLEM_PerFedAvg_ratio_e.out 2>&1 &

  nohup python -u main.py --dataset Cifar10-alpha0.5-ratio0.5-u80 --algorithm PerFedavg --batch_size 32 \
 --num_users 80 --beta 0.01 --learning_rate 0.01 --num_glob_iters 200 --local_epochs 20 --times 3 --gpu 4 > ./acc_loss_record/cifar10/local_epochs=20/PerFedAvg.out 2>&1 &


 


  