import torch
import random
import pandas as pd
import numpy as np
from copy import deepcopy


class DataSplitter():

    def __init__(self):
        self.ratings = self._load_rating()
        self._binalize()
        self.user_pool = set(self.ratings['new_uid'].unique())
        self.item_pool = set(self.ratings['new_mid'].unique())
        self.negatives = self._sample_negative()
        self.train_ratings, self.validation_ratings, self.test_ratings = self._split_data()

    def _load_rating(self):
        df = pd.read_csv(
            'data/ml-1m/ratings.dat',
            sep='::',
            header=None,
            names=['uid', 'mid', 'rating', 'timestamp'],
            engine='python'
            )

        user_id = df[['uid']].drop_duplicates()
        user_id['new_uid'] = np.arange(len(user_id))
        df = df.merge(user_id, on=['uid'])

        item_id = df[['mid']].drop_duplicates()
        item_id['new_mid'] = np.arange(len(item_id))
        df = df.merge(item_id, on=['mid'])

        df = df[['new_uid', 'new_mid', 'rating', 'timestamp']]
        return df

    def _binalize(self):
        self.ratings['rating'][self.ratings['rating'] > 0] = 1.0

    def _sample_negative(self):
        interact_status = \
            self.ratings.groupby('new_uid')['new_mid'].apply(set).reset_index().rename(columns={'new_mid': 'interacted_items'})
        interact_status['negative_items'] = \
            interact_status['interacted_items'].apply(lambda x: self.item_pool - x)
        interact_status['negative_samples_for_validation'] = \
            interact_status['negative_items'].apply(lambda x: random.sample(x, 99))
        interact_status['negative_samples_for_test'] = \
            interact_status['negative_items'].apply(lambda x: random.sample(x, 99))
        return interact_status[['new_uid', 'negative_items', 'negative_samples_for_validation', 'negative_samples_for_test']]

    def _split_data(self):
        self.ratings['timestamp_rank'] = \
            self.ratings.groupby(['new_uid'])['timestamp'].rank(method='first', ascending=False)
        test = self.ratings[self.ratings['timestamp_rank'] == 1]
        validation = self.ratings[self.ratings['timestamp_rank'] == 2]
        train = self.ratings[self.ratings['timestamp_rank'] > 2]
        return train[['new_uid', 'new_mid', 'rating']], validation[['new_uid', 'new_mid', 'rating']], test[['new_uid', 'new_mid', 'rating']]

    def make_evaluation_data(self, type):
        if type == 'test':
            ratings = pd.merge(self.test_ratings, self.negatives[['new_uid', 'negative_samples_for_test']], on='new_uid')
            ratings = ratings.rename(columns={'negative_samples_for_test': 'negative_samples'})
        elif type == 'validation':
            ratings = pd.merge(self.validation_ratings, self.negatives[['new_uid', 'negative_samples_for_validation']], on='new_uid')
            ratings = ratings.rename(columns={'negative_samples_for_validation': 'negative_samples'})
        users, items, negative_users, negative_items = [], [], [], []
        for row in ratings.itertuples():
            users.append(int(row.new_uid))
            items.append(int(row.new_mid))
            for i in range(len(row.negative_samples)):
                negative_users.append(int(row.new_uid))
                negative_items.append(int(row.negative_samples[i]))
        return [torch.LongTensor(users), torch.LongTensor(items), torch.LongTensor(negative_users), torch.LongTensor(negative_items)]
