  nohup python -u main.py --dataset FEMNIST --algorithm FedAvg --batch_size 32 \
--num_users 271 --learning_rate 0.01 --num_glob_iters 500 --local_epochs 10 --times 1 --gpu 0 > FedAvg.out 2>&1 &

  nohup python -u main.py --dataset FEMNIST --algorithm FedKLEM --batch_size 32 \
--num_users 271 --learning_rate 0.01 --num_glob_iters 500 --local_epochs 10 --times 1 --p 3 --gpu 1 > FedKLEM.out 2>&1 &

  nohup python -u main.py --dataset FEMNIST --algorithm FedKLEM_PerFedAvg --batch_size 32 \
 --num_users 271 --beta 0.01 --learning_rate 0.01 --num_glob_iters 500 --local_epochs 10 --times 1 --p 3 --gpu 2 > FedSEM.out 2>&1 &

  nohup python -u main.py --dataset FEMNIST --algorithm FedIFCA --batch_size 32 \
--num_users 271 --learning_rate 0.01 --num_glob_iters 500 --local_epochs 10 --times 1 --p 3 --gpu 3 > FedIFCA.out 2>&1 &

  nohup python -u main.py --dataset FEMNIST --algorithm FedSEM --batch_size 32 \
--num_users 271 --learning_rate 0.01 --num_glob_iters 500 --local_epochs 10 --times 1 --p 3 --gpu 4 > FedKLEM_PerFedAvg.out 2>&1 &

  nohup python -u main.py --dataset FEMNIST --algorithm PerFedavg --batch_size 32 \
 --num_users 271 --beta 0.01 --learning_rate 0.01 --num_glob_iters 500 --local_epochs 10 --times 1 --gpu 5 > PerFedAvg.out 2>&1 &
