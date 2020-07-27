import configparser

import data
from MF import MF


def main():
    config = configparser.ConfigParser()
    config.read('MF_PyTorch/config.ini')

    data_splitter = data.DataSplitter()
    validation_data = data_splitter.make_evaluation_data('validation')

    mf = MF(data_splitter.n_user, data_splitter.n_item, 20, 0.001, 0)
    print(mf)

    for epoch in range(config.getint('MODEL', 'epoch')):
        train_loader = data_splitter.make_train_loader(config.getint('MODEL', 'n_negative'), 1024)


if __name__ == "__main__":
    main()