# Multi-Initial-Center Federated Learning with Data Distribution Similarity-Aware Constraint	

Research code that accompanies the paper [Multi-Initial-Center Federated Learning with Data Distribution Similarity-Aware Constraint	).
It contains implementation of the following algorithms:
* **FedDSMIC** (the proposed algorithm) ([code](https://github.com/zhuangdizhu/FedGen/blob/main/FLAlgorithms/servers/serverFedDSMIC.py)).
* **FedAvg** ([paper](https://arxiv.org/pdf/1602.05629.pdf) and [code](https://github.com/zhuangdizhu/FedGen/blob/main/FLAlgorithms/servers/serveravg.py)).
* **FedProx** ([paper](https://arxiv.org/pdf/1812.06127.pdf) and [code](https://github.com/zhuangdizhu/FedGen/blob/main/FLAlgorithms/servers/serverFedProx.py)).
* **IFCA** ([paper](https://arxiv.org/pdf/1812.06127.pdf) and [code](https://github.com/zhuangdizhu/FedGen/blob/main/FLAlgorithms/servers/serverIFCA.py)).
* **FedSEM** ([paper](https://arxiv.org/pdf/1812.06127.pdf) and [code](https://github.com/zhuangdizhu/FedGen/blob/main/FLAlgorithms/servers/serverFedSEM.py)).
* **Per-FedAvg** ([paper](https://arxiv.org/pdf/1812.06127.pdf) and [code](https://github.com/zhuangdizhu/FedGen/blob/main/FLAlgorithms/servers/serverFedSEM.py)).

## Install Requirements:
```pip3 install -r requirements.txt```

  
## Prepare Dataset: 
* To generate *non-iid* **Mnist** Dataset following the Dirichlet distribution D(&alpha;=0.1) for 20 clients, using 50% of the total available training samples:
<pre><code>cd FedDSMIC/data/Mnist
python generate_data.py  --n_class 10 --sampling_ratio 0.5 --alpha 1.0 --n_user 10
``
### This will generate a dataset located at FedGen/data/Mnist/u20c10-alpha0.1-ratio0.5/
</code></pre>
    

- Similarly, to generate *non-iid* **EMnist** Dataset, using 10% of the total available training samples:
<pre><code>cd FedGen/data/EMnist
python generate_niid_dirichlet.py --sampling_ratio 0.1 --alpha 0.1 --n_user 20 
### This will generate a dataset located at FedGen/data/EMnist/u20-letters-alpha0.1-ratio0.1/
</code></pre> 

## Run Experiments: 

There is a main file "main.py" which allows running all experiments.

#### Run experiments on the *Mnist* Dataset:
```

