  nohup python -u main.py --dataset EMnist-alpha10.0-ratio1.0-u100 --algorithm FedAvg \
  --batch_size 32 --num_users 80 --learning_rate 0.01 --num_glob_iters 300 --E 1 --times 1 --gpu 0 > ./acc_loss_record/emnist/emnist-alpha10/E=1/FedAvg.out 2>&1 &

  nohup python -u main.py --dataset EMnist-alpha10.0-ratio1.0-u100 --algorithm FedKLEM \
  --batch_size 32 --num_users 80 --learning_rate 0.01 --num_glob_iters 300 --E 1 --times 1 --gpu 1 > ./acc_loss_record/emnist/emnist-alpha10/E=1/FedKLEM.out 2>&1 &

  nohup python -u main.py --dataset EMnist-alpha10.0-ratio1.0-u100 --algorithm FedSEM \
  --batch_size 32 --num_users 80 --learning_rate 0.01 --num_glob_iters 300 --E 1 --times 1 --gpu 2 > ./acc_loss_record/emnist/emnist-alpha10/E=1/FedSEM.out 2>&1 &

  nohup python -u main.py --dataset EMnist-alpha10.0-ratio1.0-u100 --algorithm FedIFCA \
  --batch_size 32 --num_users 80 --learning_rate 0.01 --num_glob_iters 300 --E 1 --times 1 --gpu 3 > ./acc_loss_record/emnist/emnist-alpha10/E=1/FedIFCA.out 2>&1 &

  nohup python -u main.py --dataset EMnist-alpha10.0-ratio1.0-u100 --algorithm FedKLEM_PerFedAvg\
  --batch_size 32 --beta 0.03 --num_users 80 --learning_rate 0.01 --num_glob_iters 300 --E 1 --times 1 --gpu 4 > ./acc_loss_record/emnist/emnist-alpha10/E=1/FedKLEM_PerFedAvg.out 2>&1 &

  nohup python -u main.py --dataset EMnist-alpha10.0-ratio1.0-u100 --algorithm PerFedavg\
  --batch_size 32 --beta 0.01 --num_users 80 --learning_rate 0.01 --num_glob_iters 300 --E 1 --times 1 --gpu 5 > ./acc_loss_record/emnist/emnist-alpha10/E=1/PerFedAvg.out 2>&1 &
