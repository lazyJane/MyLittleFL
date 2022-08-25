  nohup python -u main.py --dataset Mnist-alpha0.5-ratio1.0-u100 --algorithm FedAvg \
  --batch_size 32 --num_users 80 --learning_rate 0.01 --num_glob_iters 500 --E 1 --times 1 --gpu 0 > FedAvg.out 2>&1 &

  nohup python -u main.py --dataset Mnist-alpha0.5-ratio1.0-u100 --algorithm FedKLEM \
  --batch_size 32 --num_users 80 --learning_rate 0.01 --num_glob_iters 500 --E 1 --times 1 --gpu 1 > FedKLEM.out 2>&1 &

  nohup python -u main.py --dataset Mnist-alpha0.5-ratio1.0-u100 --algorithm FedSEM \
  --batch_size 32 --num_users 80 --learning_rate 0.01 --num_glob_iters 500 --E 1 --times 1 --gpu 2 > FedSEM.out 2>&1 &

  nohup python -u main.py --dataset Mnist-alpha0.5-ratio1.0-u100 --algorithm FedIFCA \
  --batch_size 32 --num_users 80 --learning_rate 0.01 --num_glob_iters 500 --E 1 --times 1 --gpu 3 > FedIFCA.out 2>&1 &

  nohup python -u main.py --dataset Mnist-alpha0.5-ratio1.0-u100 --algorithm FedKLEM_PerFedAvg\
  --batch_size 32 --beta 0.01 --num_users 80 --learning_rate 0.01 --num_glob_iters 500 --E 1 --times 1 --gpu 0 > ./acc_loss_record/mnist/E=1/FedKLEM_PerFedAvg.out 2>&1 &

  nohup python -u main.py --dataset Mnist-alpha0.5-ratio1.0-u100 --algorithm FedKLEM_PerFedAvg_ratio\
  --batch_size 32 --beta 0.01 --num_users 80 --learning_rate 0.01 --num_glob_iters 500 --E 1 --times 3 --gpu 1 > ./acc_loss_record/mnist/E=1/FedKLEM_PerFedAvg_ratio.out 2>&1 &

  nohup python -u main.py --dataset Mnist-alpha0.5-ratio1.0-u100 --algorithm FedKLEM_PerFedAvg_ratio_e\
  --batch_size 32 --beta 0.01 --num_users 80 --learning_rate 0.01 --num_glob_iters 500 --E 1 --times 3 --gpu 1 > ./acc_loss_record/mnist/E=1/FedKLEM_PerFedAvg_ratio_e.out 2>&1 &

  nohup python -u main.py --dataset Mnist-alpha0.5-ratio1.0-u100 --algorithm PerFedavg\
  --batch_size 32 --beta 0.01 --num_users 80  --learning_rate 0.01 --num_glob_iters 500 --E 1 --times 3 --gpu 1 > ./acc_loss_record/mnist/E=1/PerFedAvg.out 2>&1 &

    nohup python -u main.py --dataset Mnist-alpha0.5-ratio1.0-u100 --algorithm FedSEM \
  --batch_size 32 --num_users 80 --learning_rate 0.01 --num_glob_iters 500 --E 1 --times 3 --gpu 2 > ./acc_loss_record/mnist/E=1/FedSEM.out 2>&1 &

 