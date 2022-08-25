  nohup python -u main.py --dataset Mnist-alpha10.0-ratio1.0-u100 --algorithm FedAvg \
  --batch_size 32 --num_users 80 --learning_rate 0.01 --num_glob_iters 500 --E 3 --times 3 --gpu 0 > ./acc_loss_record/mnist/alpha=10.0/E=3/FedAvg.out 2>&1 &

  nohup python -u main.py --dataset Mnist-alpha10.0-ratio1.0-u100 --algorithm Fedlocal \
  --batch_size 32 --num_users 80 --learning_rate 0.01 --num_glob_iters 500 --E 3 --times 3 --gpu 0 > ./acc_loss_record/mnist/alpha=10.0/E=3/Fedlocal.out 2>&1 &

  nohup python -u main.py --dataset Mnist-alpha10.0-ratio1.0-u100 --algorithm FedProx \
  --batch_size 32 --num_users 80 --learning_rate 0.01 --num_glob_iters 500 --E 3 --times 3 --gpu 2 > ./acc_loss_record/mnist/alpha=10.0/E=3/FedProx.out 2>&1 &

  nohup python -u main.py --dataset Mnist-alpha10.0-ratio1.0-u100 --algorithm FedKLEM \
  --batch_size 32 --num_users 80 --learning_rate 0.01 --num_glob_iters 500 --E 3 --times 3 --gpu 5 > ./acc_loss_record/mnist/alpha=10.0/E=3/FedKLEM.out 2>&1 &

  nohup python -u main.py --dataset Mnist-alpha10.0-ratio1.0-u100 --algorithm FedSEM \
  --batch_size 32 --num_users 80 --learning_rate 0.01 --num_glob_iters 500 --E 3 --times 3 --gpu 5 > ./acc_loss_record/mnist/alpha=10.0/E=3/FedSEM.out 2>&1 &

  nohup python -u main.py --dataset Mnist-alpha10.0-ratio1.0-u100 --algorithm FedIFCA \
  --batch_size 32 --num_users 80 --learning_rate 0.01 --num_glob_iters 500 --E 3 --times 3 --gpu 5 > ./acc_loss_record/mnist/alpha=10.0/E=3/FedIFCA.out 2>&1 &

  nohup python -u main.py --dataset Mnist-alpha10.0-ratio1.0-u100 --algorithm FedKLEM_PerFedAvg\
  --batch_size 32 --beta 0.01 --num_users 80 --learning_rate 0.01 --num_glob_iters 500 --E 3 --times 3 --gpu 0 > ./acc_loss_record/mnist/alpha=10.0/E=3/FedKLEM_PerFedAvg.out 2>&1 &

    nohup python -u main.py --dataset Mnist-alpha10.0-ratio1.0-u100 --algorithm FedKLEM_PerFedAvg_ratio\
  --batch_size 32 --beta 0.01 --num_users 80 --learning_rate 0.01 --num_glob_iters 500 --E 3 --times 3 --gpu 0 > ./acc_loss_record/mnist/alpha=10.0/E=3/FedKLEM_PerFedAvg_ratio.out 2>&1 &

      nohup python -u main.py --dataset Mnist-alpha10.0-ratio1.0-u100 --algorithm FedKLEM_PerFedAvg_ratio_e\
  --batch_size 32 --beta 0.01 --num_users 80 --learning_rate 0.01 --num_glob_iters 500 --E 3 --times 3 --gpu 2 > ./acc_loss_record/mnist/alpha=10.0/E=3/FedKLEM_PerFedAvg_ratio_e.out 2>&1 &

  nohup python -u main.py --dataset Mnist-alpha10.0-ratio1.0-u100 --algorithm PerFedavg\
  --batch_size 32 --beta 0.01 --num_users 80  --learning_rate 0.01 --num_glob_iters 500 --E 3 --times 3 --gpu 2 > ./acc_loss_record/mnist/alpha=10.0/E=3/PerFedAvg.out 2>&1 &

 