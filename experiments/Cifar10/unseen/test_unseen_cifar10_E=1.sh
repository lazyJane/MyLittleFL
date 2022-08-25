nohup python -u test_unseen.py --dataset Cifar10-alpha0.5-ratio0.5-u80 --algorithm FedAvg --batch_size 32 \
--test_unseen_user_E 1 --learning_rate 0.01  --E 1 --gpu 0 > ./acc_loss_record/cifar10/E=1/unseen/FedAvg_unseen.out 2>&1 &

 nohup python -u test_unseen.py --dataset Cifar10-alpha0.5-ratio0.5-u80 --algorithm FedKLEM --batch_size 32 \
 --test_unseen_user_E 1 --learning_rate 0.01 --num_glob_iters 200  --E 1 --times 1 --gpu 1 > ./acc_loss_record/cifar10/E=1/unseen/FedKLEM_unseen.out 2>&1 &

 nohup python -u test_unseen.py --dataset Cifar10-alpha0.5-ratio0.5-u80 --algorithm FedIFCA --batch_size 32 \
 --test_unseen_user_E 1 --learning_rate 0.01 --num_glob_iters 200  --E 1 --times 1 --gpu 2 > ./acc_loss_record/cifar10/E=1/unseen/FedIFCA_unseen.out 2>&1 &

 nohup python -u test_unseen.py --dataset Cifar10-alpha0.5-ratio0.5-u80 --algorithm FedKLEM_PerFedAvg --batch_size 32 \
 --test_unseen_user_E 1 --beta 0.01 --learning_rate 0.01 --num_glob_iters 200  --E 1 --times 1 --gpu 3 > ./acc_loss_record/cifar10/E=1/unseen/FedKLEM_PerFedAvg_unseen.out 2>&1 &

 nohup python -u test_unseen.py --dataset Cifar10-alpha0.5-ratio0.5-u80 --algorithm FedSEM --batch_size 32 \
--test_unseen_user_E 1 --learning_rate 0.01 --num_glob_iters 200  --E 1 --times 1 --gpu 4 > ./acc_loss_record/cifar10/E=1/unseen/FedSEM_unseen.out 2>&1 &

 #nohup python -u test_unseen.py --dataset Cifar10-alpha0.5-ratio0.5-u80 --algorithm PerFedAvg --batch_size 32 \
 #--num_users 64 --beta 0.01 --learning_rate 0.01 --num_glob_iters 200  --E 1 --times 1 --gpu 5 > ./acc_loss_record/cifar10/E=1/unseen/PerFedAvg_unseen.out 2>&1 &


  