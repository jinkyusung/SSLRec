import pickle
import urllib.request
import zipfile
import os
import pandas as pd
import torch
from typing import Tuple
import numpy as np
from scipy.sparse import coo_matrix


def message(func):
    def wrapper(*args, **kwargs):
        print(f'Start to download MovieLens Data.')
        result = func(*args, **kwargs)
        print('Done.')
        return result
    return wrapper


def load(prefix, name):
    path = f'./datasets/general_cf/sparse_{prefix}/{name}_mat.pkl'
    with open(path, 'rb') as file:
        return pickle.load(file)
    

def save(prefix, name, sparse_matrix):
    path = f'./datasets/general_cf/sparse_{prefix}/{name}_mat.pkl'
    with open(path, 'wb') as f:
        pickle.dump(sparse_matrix, f)


class MovieLens:
    def __init__(self, tag='ml-1m', ratio=[0.8, 0.1, 0.1], random_seed=None):
        self.tag = tag
        self.ratio = ratio
        self.random_seed= random_seed

        self.url = f'http://files.grouplens.org/datasets/movielens/{tag}.zip'
        self.dataset_path = f'./{tag}.zip'
        self.extract_path = f'./{tag}'

        self.df = self.__get_dataframe()
        self.train_mask, self.val_mask, self.test_mask = self.__get_split_mask(n=len(self.df))

    @message
    def __get_dataframe(self):
        # Download and Unzip.
        urllib.request.urlretrieve(self.url, self.dataset_path)
        with zipfile.ZipFile(self.dataset_path, 'r') as zip_ref:
            zip_ref.extractall(self.extract_path)

        # Read as CSV format.
        ratings_file = os.path.join(self.extract_path, self.tag, 'ratings.dat')
        df = pd.read_csv(ratings_file, sep='::', names=["user", "item", "rating", "timestamp"], engine='python')

        return df
    
    def __get_split_mask(self, n) :
        indices = np.arange(n)
        if self.random_seed is not None:
            np.random.seed(self.random_seed)
        np.random.shuffle(indices)
        
        train_size = int(self.ratio[0] * n)
        val_size = int(self.ratio[1] * n)
        
        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size + val_size]
        test_indices = indices[train_size + val_size:]
        
        train_mask = np.zeros(n, dtype=bool)
        val_mask = np.zeros(n, dtype=bool)
        test_mask = np.zeros(n, dtype=bool)
        
        train_mask[train_indices] = True
        val_mask[val_indices] = True
        test_mask[test_indices] = True
        
        return train_mask, val_mask, test_mask

    def get_sparse_matrix(self):
        user2idx = {user: idx for idx, user in enumerate(self.df['user'].unique())}
        item2idx = {item: idx for idx, item in enumerate(self.df['item'].unique())}

        n_user = len(user2idx)
        n_item = len(item2idx)

        self.df['user'] = self.df['user'].map(user2idx)
        self.df['item'] = self.df['item'].map(item2idx)

        total_user = np.array(self.df['user'])
        total_item = np.array(self.df['item'])
        
        train_user = total_user[self.train_mask]
        train_item = total_item[self.train_mask]
        train_ones = np.ones(len(train_user))

        val_user = total_user[self.val_mask]
        val_item = total_item[self.val_mask]
        val_ones = np.ones(len(val_user))

        test_user = total_user[self.test_mask]
        test_item = total_item[self.test_mask]
        test_ones = np.ones(len(test_user))

        train_sparse_matrix = coo_matrix((train_ones, (train_user, train_item)), shape=(n_user, n_item), dtype=float)
        val_sparse_matrix = coo_matrix((val_ones, (val_user, val_item)), shape=(n_user, n_item), dtype=float)
        test_sparse_matrix = coo_matrix((test_ones, (test_user, test_item)), shape=(n_user, n_item), dtype=float)

        return train_sparse_matrix, val_sparse_matrix, test_sparse_matrix
    

if __name__ == '__main__':

    movielens = MovieLens(tag='ml-1m', ratio=[0.8, 0.1, 0.1], random_seed=42)
    
    train_sparse_matrix, valid_sparse_matrix, test_sparse_matrix = movielens.get_sparse_matrix()

    save('movielens', 'train', train_sparse_matrix)
    save('movielens', 'valid', valid_sparse_matrix)
    save('movielens', 'test', test_sparse_matrix)
