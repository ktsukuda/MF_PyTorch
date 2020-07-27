import tqdm
import configparser
from torch import nn, optim

import data
from MF import MF


def main():
    config = configparser.ConfigParser()
    config.read('MF_PyTorch/config.ini')

    data_splitter = data.DataSplitter()
    validation_data = data_splitter.make_evaluation_data('validation')

    model = MF(data_splitter.n_user, data_splitter.n_item, 20)
    print(model)
    model.to('cuda:0')
    model.train()

    opt = optim.Adam(model.parameters(), lr=0.001, weight_decay=0)
    criterion = nn.BCELoss()

    for epoch in range(config.getint('MODEL', 'epoch')):
        train_loader = data_splitter.make_train_loader(config.getint('MODEL', 'n_negative'), 1024)
        total_loss = 0
        for batch in tqdm.tqdm(train_loader):
            users, items, ratings = batch[0], batch[1], batch[2]
            users = users.to('cuda:0')
            items = items.to('cuda:0')
            ratings = ratings.to('cuda:0')
            opt.zero_grad()
            pred = model(users, items)
            loss = criterion(pred, ratings)
            loss.backward()
            opt.step()
            total_loss += loss.item()


if __name__ == "__main__":
    main()
