  nohup python -u main.py --dataset Mnist-alpha0.5-ratio1.0-u100 --algorithm FedAvg \
  --batch_size 32 --num_users 80 --learning_rate 0.01 --num_glob_iters 500 --local_epochs 5 --times 3 --gpu 0 > ./acc_loss_record/mnist/local_epochs=5/FedAvg.out 2>&1 &

  nohup python -u main.py --dataset Mnist-alpha0.5-ratio1.0-u100 --algorithm FedKLEM \
  --batch_size 32 --num_users 80 --learning_rate 0.01 --num_glob_iters 500 --local_epochs 5 --times 3 --gpu 1 > ./acc_loss_record/mnist/local_epochs=5/FedKLEM.out 2>&1 &

  nohup python -u main.py --dataset Mnist-alpha0.5-ratio1.0-u100 --algorithm FedProx --batch_size 32 \
  --num_users 80 --learning_rate 0.01 --num_glob_iters 500 --local_epochs 5 --times 3 --gpu 0 > ./acc_loss_record/mnist/local_epochs=5/FedProx.out 2>&1 &

  nohup python -u main.py --dataset Mnist-alpha0.5-ratio1.0-u100 --algorithm Fedlocal --batch_size 32 \
  --num_users 80 --learning_rate 0.01 --num_glob_iters 500 --local_epochs 5 --times 3 --gpu 1 > ./acc_loss_record/mnist/local_epochs=5/Fedlocal.out 2>&1 &

  nohup python -u main.py --dataset Mnist-alpha0.5-ratio1.0-u100 --algorithm FedSEM \
  --batch_size 32 --num_users 80 --learning_rate 0.01 --num_glob_iters 500 --local_epochs 5 --times 3 --gpu 2 > ./acc_loss_record/mnist/local_epochs=5/FedSEM.out 2>&1 &

  nohup python -u main.py --dataset Mnist-alpha0.5-ratio1.0-u100 --algorithm FedIFCA \
  --batch_size 32 --num_users 80 --learning_rate 0.01 --num_glob_iters 500 --local_epochs 5 --times 3 --gpu 3 > ./acc_loss_record/mnist/local_epochs=5/FedIFCA.out 2>&1 &

  nohup python -u main.py --dataset Mnist-alpha0.5-ratio1.0-u100 --algorithm FedKLEM_PerFedAvg\
  --batch_size 32 --beta 0.01 --num_users 80 --learning_rate 0.01 --num_glob_iters 500 --local_epochs 5 --times 3 --gpu 0 > ./acc_loss_record/mnist/local_epochs=5/FedKLEM_PerFedAvg.out 2>&1 &

  nohup python -u main.py --dataset Mnist-alpha0.5-ratio1.0-u100 --algorithm PerFedavg\
  --batch_size 32 --beta 0.01 --num_users 80  --learning_rate 0.01 --num_glob_iters 500 --local_epochs 5 --times 3 --gpu 5 > ./acc_loss_record/mnist/local_epochs=5/PerFedAvg.out 2>&1 &

 