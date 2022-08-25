  nohup python -u main.py --dataset EMnist-alpha0.5-ratio1.0-u100 --algorithm FedAvg \
  --batch_size 32 --num_users 16 --learning_rate 0.01 --num_glob_iters 300 --local_epochs 10 --times 1 --gpu 0 > ./acc_loss_record/emnist/emnist-alpha0.5/r=0.2/local_epochs=10/FedAvg.out 2>&1 &

  nohup python -u main.py --dataset EMnist-alpha0.5-ratio1.0-u100 --algorithm FedKLEM \
  --batch_size 32 --num_users 16 --learning_rate 0.01 --num_glob_iters 300 --local_epochs 10 --times 1 --gpu 1 --p 6 > ./acc_loss_record/emnist/emnist-alpha0.5/r=0.2/local_epochs=10/FedKLEM.out 2>&1 &

  nohup python -u main.py --dataset EMnist-alpha0.5-ratio1.0-u100 --algorithm FedSEM \
  --batch_size 32 --num_users 16 --learning_rate 0.01 --num_glob_iters 300 --local_epochs 10 --times 1 --gpu 2 --p 6 > ./acc_loss_record/emnist/emnist-alpha0.5/r=0.2/local_epochs=10/FedSEM.out 2>&1 &

  nohup python -u main.py --dataset EMnist-alpha0.5-ratio1.0-u100 --algorithm FedIFCA \
  --batch_size 32 --num_users 16 --learning_rate 0.01 --num_glob_iters 300 --local_epochs 10 --times 1 --gpu 5 --p 6 > ./acc_loss_record/emnist/emnist-alpha0.5/r=0.2/local_epochs=10/FedIFCA.out 2>&1 &

  nohup python -u main.py --dataset EMnist-alpha0.5-ratio1.0-u100 --algorithm FedKLEM_PerFedAvg\
  --batch_size 32 --beta 0.01 --num_users 16 --learning_rate 0.01 --num_glob_iters 300 --local_epochs 10 --times 1 --gpu 0 --p 6 > ./acc_loss_record/emnist/emnist-alpha0.5/r=0.2/local_epochs=10/FedKLEM_PerFedAvg.out 2>&1 &

    nohup python -u main.py --dataset EMnist-alpha0.5-ratio1.0-u100 --algorithm FedKLEM_PerFedAvg_ratio\
  --batch_size 32 --beta 0.01 --num_users 16 --learning_rate 0.01 --num_glob_iters 300 --local_epochs 10 --times 1 --gpu 2 --p 6 > ./acc_loss_record/emnist/emnist-alpha0.5/r=0.2/local_epochs=10/FedKLEM_PerFedAvg_ratio.out 2>&1 &

      nohup python -u main.py --dataset EMnist-alpha0.5-ratio1.0-u100 --algorithm FedKLEM_PerFedAvg_ratio_e\
  --batch_size 32 --beta 0.01 --num_users 16 --learning_rate 0.01 --num_glob_iters 300 --local_epochs 10 --times 1 --gpu 3 --p 6 > ./acc_loss_record/emnist/emnist-alpha0.5/r=0.2/local_epochs=10/FedKLEM_PerFedAvg_ratio_e.out 2>&1 &

  nohup python -u main.py --dataset EMnist-alpha0.5-ratio1.0-u100 --algorithm PerFedavg\
  --batch_size 32 --beta 0.01 --num_users 16 --learning_rate 0.01 --num_glob_iters 300 --local_epochs 10 --times 1 --gpu 4 > ./acc_loss_record/emnist/emnist-alpha0.5/r=0.2/local_epochs=10/PerFedAvg.out 2>&1 &

  nohup python -u main.py --dataset EMnist-alpha0.5-ratio1.0-u100 --algorithm FedProx \
  --batch_size 32 --num_users 40 --learning_rate 0.01 --num_glob_iters 300 --local_epochs 10 --times 1 --gpu 2 > ./acc_loss_record/emnist/emnist-alpha0.5/r=0.2/local_epochs=10/FedProx.out 2>&1 &

  nohup python -u main.py --dataset EMnist-alpha0.5-ratio1.0-u100 --algorithm Fedlocal \
  --batch_size 32 --num_users 40 --learning_rate 0.01 --num_glob_iters 300 --local_epochs 10 --times 1 --gpu 3 > ./acc_loss_record/emnist/emnist-alpha0.5/r=0.2/local_epochs=10/Fedlocal.out 2>&1 &