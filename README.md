# MF_PyTorch

Matrix Factorization with PyTorch.

## Environment

- Python: 3.6
- PyTorch: 1.5.1
- CUDA: 10.1
- Python library: see [conda_requirements.txt](https://github.com/ktsukuda/MF_PyTorch/blob/master/conda_requirements.txt)

## Dataset

[The Movielens 1M Dataset](http://grouplens.org/datasets/movielens/1m/) is used. The rating data is included in [data/ml-1m](https://github.com/ktsukuda/MF_PyTorch/tree/master/data/ml-1m).

## Getting Started

1. Clone the repository and install requirements in a specific conda environment

```bash
$ git clone https://github.com/ktsukuda/MF_PyTorch.git
$ cd MF_PyTorch
$ conda create --name <env> --file conda_requirements.txt
$ conda activate <env>
```

2. Run model with build-in dataset

```bash
$ python MF_PyTorch/main.py
```

## Details

For each user, the latest and the second latest rating are used as test and validation, respectively. The remaining ratings are used as training. The hyperparameters (batch_size, lr, latent_dim, l2_reg) are tuned by using the valudation data in terms of nDCG. See [config.ini](https://github.com/ktsukuda/MF_PyTorch/blob/master/MF_PyTorch/config.ini) about the range of each hyperparameter.

Although the original ratings range 1 to 5, all of them are converted to 1. That is, we use the binalized data where movies rated by users have score 1 while those not rated by users have score 0.

By running the code, hyperparameters are automatically tuned. After the training process, the best hyperparameters and the best nDCG computed by using the test data are displayed.

Given a specific combination of hyperparameters, the corresponding training results are saved in `data/train_result/<hyperparameter combination>` (e.g., data/train_result/batch_size_512-lr_0.005-latent_dim_8-l2_reg_1e-07-epoch_3-n_negative_4-top_k_10). In the directory, a model file (`model.pth`) and a json file (`epoch_data.json`) that describes information for each epoch are generated. The json file can be described as follows (epoch=3).

```json
[
    {
        "epoch": 0,
        "loss": 3275.6108826696873,
        "HR": 0.4460264900662252,
        "NDCG": 0.2433340828186714
    },
    {
        "epoch": 1,
        "loss": 1510.2559289187193,
        "HR": 0.6197019867549669,
        "NDCG": 0.3502363951558794
    },
    {
        "epoch": 2,
        "loss": 1320.9795952737331,
        "HR": 0.6700331125827814,
        "NDCG": 0.3889819661175262
    }
]
```
