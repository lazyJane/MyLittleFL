nohup python -u test_unseen.py --dataset Cifar100-alpha0.5-ratio0.5-u125 --algorithm FedAvg --batch_size 32 \
--test_unseen_user_E 3 --learning_rate 0.01  --E 3 --gpu 0 > ./acc_loss_record/cifar100_user125_unseen25/E=3/unseen/FedAvg_unseen.out 2>&1 &

 nohup python -u test_unseen.py --dataset Cifar100-alpha0.5-ratio0.5-u125 --algorithm FedKLEM --batch_size 32 \
 --test_unseen_user_E 3 --learning_rate 0.01 --num_glob_iters 200  --E 3 --times 1 --gpu 1 > ./acc_loss_record/cifar100_user125_unseen25/E=3/unseen/FedKLEM_unseen.out 2>&1 &

 nohup python -u test_unseen.py --dataset Cifar100-alpha0.5-ratio0.5-u125 --algorithm FedIFCA --batch_size 32 \
 --test_unseen_user_E 3 --learning_rate 0.01 --num_glob_iters 200  --E 3 --times 1 --gpu 2 > ./acc_loss_record/cifar100_user125_unseen25/E=3/unseen/FedIFCA_unseen.out 2>&1 &

 nohup python -u test_unseen.py --dataset Cifar100-alpha0.5-ratio0.5-u125 --algorithm FedKLEM_PerFedAvg --batch_size 32 \
 --test_unseen_user_E 3 --beta 0.01 --learning_rate 0.01 --num_glob_iters 200  --E 3 --times 1 --gpu 3 > ./acc_loss_record/cifar100_user125_unseen25/E=3/unseen/FedKLEM_PerFedAvg_unseen.out 2>&1 &

  nohup python -u test_unseen.py --dataset Cifar100-alpha0.5-ratio0.5-u125 --algorithm FedKLEM_PerFedAvg_ratio --batch_size 32 \
 --test_unseen_user_E 3 --beta 0.01 --learning_rate 0.01 --num_glob_iters 200  --E 3 --times 1 --gpu 4 > ./acc_loss_record/cifar100_user125_unseen25/E=3/unseen/FedKLEM_PerFedAvg_ratio_unseen.out 2>&1 &

  nohup python -u test_unseen.py --dataset Cifar100-alpha0.5-ratio0.5-u125 --algorithm FedKLEM_PerFedAvg_ratio_e --batch_size 32 \
 --test_unseen_user_E 3 --beta 0.01 --learning_rate 0.01 --num_glob_iters 200  --E 3 --times 1 --gpu 5 > ./acc_loss_record/cifar100_user125_unseen25/E=3/unseen/FedKLEM_PerFedAvg_ratio_e_unseen.out 2>&1 &

 nohup python -u test_unseen.py --dataset Cifar100-alpha0.5-ratio0.5-u125 --algorithm FedSEM --batch_size 32 \
--test_unseen_user_E 3 --learning_rate 0.01 --num_glob_iters 200  --E 3 --times 1 --gpu 0 > ./acc_loss_record/cifar100_user125_unseen25/E=3/unseen/FedSEM_unseen.out 2>&1 &

 nohup python -u test_unseen.py --dataset Cifar100-alpha0.5-ratio0.5-u125 --algorithm PerFedavg --batch_size 32 \
--test_unseen_user_E 3 --beta 0.01 --learning_rate 0.01 --num_glob_iters 200  --E 3 --times 1 --gpu 3 > ./acc_loss_record/cifar100_user125_unseen25/E=3/unseen/PerFedAvg_unseen.out 2>&1 &


  