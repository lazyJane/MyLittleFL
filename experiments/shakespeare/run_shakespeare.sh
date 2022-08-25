python main.py --dataset shakespeare --algorithm FedAvg --batch_size 32 \
--num_users 112 --learning_rate 0.01 --num_glob_iters 500 --E 1 --times 1

python main.py --dataset shakespeare --algorithm FedKLEM --batch_size 32 \
--num_users 112 --learning_rate 0.01 --num_glob_iters 500 --E 1 --times 1

python main.py --dataset shakespeare --algorithm FedIFCA --batch_size 32 \
--num_users 112 --learning_rate 0.01 --num_glob_iters 500 --E 1 --times 1

python main.py --dataset shakespeare --algorithm PerFedavg --batch_size 32 \
--num_users 112 --learning_rate 0.01 --num_glob_iters 500 --E 1 --times 1

python main.py --dataset shakespeare  --algorithm FedProx --batch_size 32 \
--num_users 20 --learning_rate 0.01 --num_glob_iters 500 --E 1 --times 1

python main.py --dataset shakespeare --algorithm Fedlocal --batch_size 32 \
--num_users 20 --learning_rate 0.01 --num_glob_iters 500 --E 1 --times 1

python main.py --dataset FEMNIST --algorithm FedKLtopk --batch_size 32 \
--num_users 20 --learning_rate 0.01 --num_glob_iters 500 --E 1 --times 1

