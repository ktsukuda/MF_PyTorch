import os
import json
import tqdm
import torch
import configparser
from torch import nn, optim

import data
import evaluation
from MF import MF


def train(model, opt, criterion, data_splitter, validation_data, config):
    epoch_data = []
    for epoch in range(config.getint('MODEL', 'epoch')):
        model.train()
        train_loader = data_splitter.make_train_loader(config.getint('MODEL', 'n_negative'), 1024)
        total_loss = 0
        for batch in tqdm.tqdm(train_loader):
            users, items, ratings = batch[0], batch[1], batch[2].float()
            users = users.to('cuda:0')
            items = items.to('cuda:0')
            ratings = ratings.to('cuda:0')
            opt.zero_grad()
            pred = model(users, items)
            loss = criterion(pred.view(-1), ratings)
            loss.backward()
            opt.step()
            total_loss += loss.item()
        hit_ratio, ndcg = evaluation.evaluate(model, validation_data, config.getint('EVALUATION', 'top_k'))
        epoch_data.append({'epoch': epoch, 'loss': total_loss, 'HR': hit_ratio, 'NDCG': ndcg})
        print('[Epoch {}] Loss = {:.2f}, HR = {:.4f}, NDCG = {:.4f}'.format(epoch, total_loss, hit_ratio, ndcg))
    return epoch_data


def save_train_result(model, epoch_data, batch_size, lr, latent_dim, l2_reg, config):
    result_dir = "data/train_result/batch_size_{}-lr_{}-latent_dim_{}-l2_reg_{}-epoch_{}-n_negative_{}-top_k_{}".format(
        batch_size, lr, latent_dim, l2_reg, config['MODEL']['epoch'], config['MODEL']['n_negative'], config['EVALUATION']['top_k'])
    os.makedirs(result_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(result_dir, 'model.pth'))
    with open(os.path.join(result_dir, 'epoch_data.json'), 'w') as f:
        json.dump(epoch_data, f, indent=4)


def main():
    config = configparser.ConfigParser()
    config.read('MF_PyTorch/config.ini')

    data_splitter = data.DataSplitter()
    validation_data = data_splitter.make_evaluation_data('validation')

    for batch_size in map(int, config['MODEL']['batch_size'].split()):
        for lr in map(float, config['MODEL']['lr'].split()):
            for latent_dim in map(int, config['MODEL']['latent_dim'].split()):
                for l2_reg in map(float, config['MODEL']['l2_reg'].split()):
                    print('batch_size = {}, lr = {}, latent_dim = {}, l2_reg = {}'.format(
                        batch_size, lr, latent_dim, l2_reg))
                    model = MF(data_splitter.n_user, data_splitter.n_item, latent_dim)
                    model.to('cuda:0')

                    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=l2_reg)
                    criterion = nn.BCELoss()
                    epoch_data = train(model, opt, criterion, data_splitter, validation_data, config)
                    save_train_result(model, epoch_data, batch_size, lr, latent_dim, l2_reg, config)


if __name__ == "__main__":
    main()
