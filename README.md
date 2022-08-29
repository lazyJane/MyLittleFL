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
modelling (Shakespeare), in addition to a synthetic dataset

Shakespeare dataset (resp. FEMNIST) was naturally partitioned by assigning
all lines from the same characters (resp. all images from the same writer)
to the same client.  We created federated versions of CIFAR10 and EMNIST by
distributing samples with the same label across the clients according to a 
symmetric Dirichlet distribution with parameter 0.4. For CIFAR100,
we exploited the availability of "coarse" and "fine" labels, using a two-stage
Pachinko allocation method  to assign 600 sample to each of the 100 clients.

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

## Prepare Dataset: 
We provide 3 non-iid settings
* To generate *non-iid* **Mnist** Dataset following the Dirichlet distribution D(&alpha;=0.1) for 20 clients, using 50% of the total available training samples:

<pre><code>cd FedDSMIC/data/Mnist
python generate_data.py  --n_class 10 --sampling_ratio 0.5 --alpha 1.0 --n_user 10
``
### This will generate a dataset located at FedDSMIC/data/Mnist/u20c10-alpha0.1-ratio0.5/
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

