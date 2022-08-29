# Multi-Initial-Center Federated Learning with Data Distribution Similarity-Aware Constraint	

Research code that accompanies the paper [Multi-Initial-Center Federated Learning with Data Distribution Similarity-Aware Constraint).
Federated Learning (FL) has recently attracted high attention since it allows clients to collaboratively train a model while the training data remains local. However, due to the inherent heterogeneity of local data distributions, the trained model usually fails to perform well on each client. Clustered FL has emerged to tackle this issue by clustering clients with similar data distributions. However, these model-dependent clustering methods tend to be costly and perform poorly. In this work, we propose a distribution similarity-based clustered federated learning framework FedDSMIC, which clusters clients by detecting the client-level underlying data distribution based on the model's memory of training data. Furthermore, we extend the assumption about data distribution to a more realistic(complicated) cluster structure. The center models are learned as good initial points to obtain common data properties in the cluster. Each client in a cluster gets a more personalized model by performing one step of gradient descent from the initial point. The empirical evaluation on real-world datasets shows that FedDSMIC outperforms popular state-of-the-art federated learning algorithms while keeping the lowest communication overhead.

It contains implementation of the following algorithms:
* **FedDSMIC** (the proposed algorithm) ([code](https://github.com/zhuangdizhu/FedGen/blob/main/FLAlgorithms/servers/serverFedDSMIC.py)).
* **FedAvg** ([paper](https://arxiv.org/pdf/1602.05629.pdf) and [code](https://github.com/zhuangdizhu/FedGen/blob/main/FLAlgorithms/servers/serveravg.py)).
* **FedProx** ([paper](https://arxiv.org/pdf/1812.06127.pdf) and [code](https://github.com/zhuangdizhu/FedGen/blob/main/FLAlgorithms/servers/serverFedProx.py)).
* **IFCA** ([paper](https://arxiv.org/pdf/1812.06127.pdf) and [code](https://github.com/zhuangdizhu/FedGen/blob/main/FLAlgorithms/servers/serverIFCA.py)).
* **FedSEM** ([paper](https://arxiv.org/pdf/1812.06127.pdf) and [code](https://github.com/zhuangdizhu/FedGen/blob/main/FLAlgorithms/servers/serverFedSEM.py)).
* **Per-FedAvg** ([paper](https://arxiv.org/pdf/1812.06127.pdf) and [code](https://github.com/zhuangdizhu/FedGen/blob/main/FLAlgorithms/servers/serverFedSEM.py)).

## Install Requirements:
```pip3 install -r requirements.txt```

## Datasets

We provide five federated benchmark datasets spanning a wide range
of machine learning tasks: image classification (CIFAR10 and CIFAR100),
handwritten character recognition (EMNIST and FEMNIST), and language
modelling (Shakespeare).

For non-iid setting, We provide 3 non-iid settings: label_swapped_non_iid_split, dirichlet_non_iid_split, pathological_non_iid_split, in addition to a iid_split.

Shakespeare dataset (resp. FEMNIST) was naturally partitioned by assigning
all lines from the same characters (resp. all images from the same writer)
to the same client.  

The following table summarizes the datasets and models

|Dataset         | Task |  Model |
| ------------------  |  ------|------- |
| FEMNIST   |     Handwritten character recognition       |     2-layer CNN + 2-layer FFN  |
| EMNIST    |    Handwritten character recognition     |      2-layer CNN + 2-layer FFN     |
| CIFAR10   |     Image classification        |      MobileNet-v2 |
| CIFAR100    |     Image classification         |      MobileNet-v2  |
| Shakespeare |     Next character prediction        |      Stacked LSTM    |
| Synthetic dataset| Binary classification | Linear model | 

See the `README.md` files of respective dataset, i.e., `data/$DATASET`,
for instructions on generating data

----
## Prepare Dataset: 

* To generate *non-iid* **Mnist** Dataset following the Dirichlet distribution D(&alpha;=0.1) for 20 clients, using 50% of the total available training samples:

```
python generate_data.py \
    --n_users 100 \
    --split dirichlet_non_iid_split\
    --n_components 3 \
    --alpha 0.5 \
    --s_frac 1.0 \
    --tr_frac 0.8 \
    --unseen_tasks_frac 0.2 \
    --seed 12345  
```

```
python generate_data.py \
    --n_users 30 \
    --split split_iid\
    --s_frac 0.02 \
    --tr_frac 0.8 \
    --seed 12345 
```

```
python generate_data.py \
    --n_users 100 \
    --split pathological_non_iid_split\
    --s_frac 1.0 \
    --tr_frac 0.8 \
    --n_shards 2 \
    --seed 12345 
```

```
python generate_data.py \
    --n_users 20 \
    --split label_swapped_non_iid_split \
    --n_components 4 \
    --s_frac 1.0 \
    --tr_frac 0.8 \
    --seed 12345  
```

    
----
## Run Experiments: 

There is a main file "main.py" which allows running all experiments.

#### Run experiments on the *Mnist* Dataset:
```
nohup python -u main.py --dataset Mnist-alpha0.5-ratio1.0-u100 --algorithm FedAvg \
  --batch_size 32 --num_users 8 --learning_rate 0.01 --num_glob_iters 500 --E 1 --times 1 --gpu 1 > ./acc_loss_record/mnist/E=1/r=0.1/FedAvg.out 2>&1 &
```
We provide example scripts to run paper experiments under experiments/ directory.

----

### Plot
For the input attribute **algorithms**, list the name of algorithms and separate them by comma, e.g. `--algorithms FedAvg,FedGen,FedProx`
```
  python main_plot.py --dataset Mnist-alpha0.5-ratio1.0-u100 --algorithms FedDSMIC,FedAvg,FedProx,PerFedavg,FedSEM,IFCA\
  --batch_size 32 --E 1 --num_users 80 --num_glob_iters 200 --plot_legend 1 --test_acc True 
```
