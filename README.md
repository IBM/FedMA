## Federated Learning with Matched Averaging
This is the code accompanying the ICLR 2020 paper "Federated Learning with Matched Averaging " Paper link: [https://openreview.net/forum?id=BkluqlSFDS]

### Overview
---
FedMA algorithm is designed for federated learning of modern neural network architectures e.g. convolutional neural networks (CNNs) and LSTMs. FedMA constructs the shared global model in a layer-wise manner by matching and averaging hidden elements (i.e. channels for convolution layers; hidden states for LSTM; neurons for fully connected layers) with similar feature extraction signatures.

### Depdendencies
---
Tested stable depdencises:
* python 3.6.5 (Anaconda)
* PyTorch 1.1.0
* torchvision 0.2.2
* CUDA 10.0.130
* cuDNN 7.5.1
* lapsolver 1.0.2

### Data Preparation
---
#### Language Models:
For the language model experiments, we used the Shakespeare dataset provided by project [Leaf](https://github.com/TalwalkarLab/leaf). Following the [instructions to prepare Shakespeare dataset](https://github.com/TalwalkarLab/leaf/tree/master/data/shakespeare), we choose to use non-i.i.d., full-size dataset, and split 80% of the data points into the training dataset. Moreover, we set minimum number of samples per user at 9K. 
Thus, the following command returns our data partitioning:
```
./preprocess.sh -s niid --sf 1.0 -k 0 -t sample -tf 0.8 -k 9
```

#### Image Classification:
We simulate a heterogeneous partition for which batch sizes and class proportions are unbalanced. We simulate a heterogeneous partition by sampling proportion of the data points in each class across participating clients from a Dirichlet distribution.  Due to the small concentration parameter (0.5) of the Dirichlet distribution, some sampled batches may not have any examples
of certain classes of data. Details about this partition can be found in the `partition_data` function in `./utils.py`.

### Experients over Language Task:
---
The source code involving language task experiments i.e. LSTM over the Shakespeare dataset locates in the folder `FedMA/language_modeling`. And we summarize the functionality of each script below.

| Script                      | Functionality                                 |
| ----------------------------- | ---------------------------------------- |
| `ensemble_accuracy_calculator.py` | Evaluating the performance of ensemble accross local models trained on paritipating clients. |
| `language_main.py`      | Conducting `FedAvg` and `FedProx` experiments, which are used as baseline methods. |
| `language_oneshot_matching.py` | Evaluating the performance of one-shot match i.e. [PFNM](https://github.com/IBM/probabilistic-federated-neural-matching)-style model fusion. |
| `language_whole_training.py` | Centralized training over one device i.e. we combine the local datasets and coduct centralized training. This is the strongest possible baseline for any Federated Leaarning method. |
| `lstm_fedma_with_comm.py` | Our proposed "FedMA with communication algorithm". |

### Experients over Image Classification Task:
---
The main result related to the image classification task i.e. VGG-9 on CIFAR-10 can be reproduced via running `./run.sh`. The following arguments to the `./main.py` file control the important parameters of the experiment.

| Argument                      | Description                                 |
| ----------------------------- | ---------------------------------------- |
| `model`                     | The CNN architecture that each client train locally. |
| `dataset`      | Dataset to use. We use CIFAR-10 to study FedMA. |
| `lr` | Inital learning rate that will be use. |
| `retrain_lr` | The learning rate for the local re-training process. Usually set to the same value as `lr` |
| `batch-size` | Batch size for the optimizers e.g. SGD or Adam. |
| `epochs` | Locally training epochs. |
| `retrain_epochs` | Local re-training epochs. |
| `n_nets` | Number of participating local clients. |
| `partition`  | Data partitioning strategy. Set to `hetero-dir` for the simulated heterogeneous CIFAR-10 dataset. |
| `comm_type`    | Federated learning methods. Set to `fedavg`, `fedprox`, or `fedma`.  |
| `comm_round`    | Number of communication rounds to use in `fedavg`, `fedprox`, and `fedma`. |
| `retrain`    | Flag to retrain the model or load from checkpoint.   |
| `rematching` | Flag to re-conduct the matching process or load from checkpoint. |

#### Sample command
```
python main.py --model=moderate-cnn \
--dataset=cifar10 \
--lr=0.01 \
--retrain_lr=0.01 \
--batch-size=64 \
--epochs=20 \
--retrain_epochs=20 \
--n_nets=16 \
--partition=hetero-dir \
--comm_type=fedma \
--comm_round=50 \
--retrain=True \
--rematching=True
```

### Interpretability of FedMA:
---
The results of interpretability we presented in the FedMA paper are summerized in a jupyter notebook i.e. `./jupyter_notebook/Interpretability_fedma.ipynb`.

### Handling Data Bias Experiments:
---
The handeling data bias experiments we presented the FedMA paper are summerized in the script `./dist_skew_main.py`. To reproduce the experiment, one can simply run:
```
bash run_dist_skew.sh
```

#### Sample command
```
python dist_skew_main.py --model=moderate-cnn \
--dataset=cifar10 \
--lr=0.01 \
--retrain_lr=0.01 \
--batch-size=64 \
--epochs=10 \
--retrain_epochs=20 \
--n_nets=2 \
--partition=homo \
--comm_type=fedma \
--retrain=True \
--rematching=True
```

### Citing FedMA:
---
```
@inproceedings{
Wang2020Federated,
title={Federated Learning with Matched Averaging},
author={Hongyi Wang and Mikhail Yurochkin and Yuekai Sun and Dimitris Papailiopoulos and Yasaman Khazaeni},
booktitle={International Conference on Learning Representations},
year={2020},
url={https://openreview.net/forum?id=BkluqlSFDS}
}
```
