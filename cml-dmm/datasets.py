import os
import numpy as np
import pandas as pd

from itertools import product
from sklearn.model_selection import train_test_split
def load_movielens(data_path='./data/ml-1m/ml-1m/ratings.dat',):
    data_path = os.path.join(data_path, 'ml-1m/ratings.dat')
    rating_cols = ['user_id', 'item_id', 'rating', 'timestamp']
    movielens = pd.read_table(data_path,
                        sep='::',
                        names=rating_cols,
                        engine='python',
                        encoding='ISO-8859-1')
    # Set user/item id and number of users/items.
    return movielens

def create_netflix_csv(file_name, data_path):   
    file_path = os.path.join(data_path, file_name)

    #read all txt file and store them in one big file
    data = open(file_path, mode='w')
    
    row = list()
    files = ['combined_data_1.txt', 'combined_data_2.txt',
            'combined_data_3.txt', 'combined_data_4.txt']
    files = [os.path.join(data_path, file) for file in files]
    for file in files:
        #print('reading ratings from {}...'.format(file))
        with open(file) as f:
            for line in f:
                del row[:]
                line = line.strip()
                if line.endswith(':'):
                    #all are rating
                    movid_id = line.replace(':', '')
                else:
                    row = [x for x in line.split(',')]
                    row.insert(0, movid_id)
                    data.write(','.join(row))
                    data.write('\n')
        #print('Done.\n')
    data.close()

def create_df(file_name="data.csv", data_path='./data/netfilx'):
    # Read all data into a pd dataframe
    file_path = os.path.join(data_path, file_name)
    df = pd.read_csv(file_path, names=['item_id','user_id','rating','timestamp'], usecols=[0, 1, 2, 3])    
    df = df.reindex(columns=['item_id','user_id','rating','timestamp'])
    return df.reset_index(drop=True)

def filter_netfilx(df=None, user_min=10, item_min=10):
    if df is None:
        return 

    user_counts = df.groupby('user_id').size()
    user_subset = np.in1d(df.user_id,user_counts[user_counts >= item_min].sample(10000).index)
    
    filter_df = df[user_subset].reset_index(drop=True)
    
    # find items with 10 or more users
    item_counts = filter_df.groupby('item_id').size()
    item_subset = np.in1d(filter_df.item_id,item_counts[item_counts >= user_min].sample(5000).index)    
    
    filter_df = filter_df[item_subset].reset_index(drop=True)
    
    # cannot have user ids with less than 5...
    user_counts = filter_df.groupby('user_id').size()
    user_subset = np.in1d(filter_df.user_id,user_counts[user_counts >= 5].index)
    
    filter_df = filter_df[user_subset].reset_index(drop=True)
    
    assert (filter_df.groupby('user_id').size() < 5).sum() == 0
    assert (filter_df.groupby('item_id').size() < 5).sum() == 0
    
    
    #print(filter_df.nunique())
    #print(filter_df.shape)
    
    return filter_df

def preprocess_netflix(file_name, data_path):
    create_netflix_csv(file_name, data_path)
    df = create_df(file_name, data_path)
    df = filter_netfilx(df, user_min=10, item_min=10)

    file_path = os.path.join(data_path, file_name)
    df.to_csv(file_path, index=False)

    return df
def load_netflix(data_path='./data/netflix', file_name='data.csv', ):
    file_path = os.path.join(data_path, file_name)
    if os.path.exists(file_path):
        data = pd.read_csv(file_path, sep=',', 
                           names=['item_id','user_id','rating','timestamp'])
    else:
        data = preprocess_netflix(file_name, data_path)
    return data

def load_amazon_music(data_path='./data/amazon-digital-music'):
    data_path = os.path.join('./data', 'amazon-digital-music/train.csv')
    df = pd.read_csv(data_path)
    return df

def load_pinterest(data_path):
    raise NotImplementedError
    
def cartesian_product(*arrays):
    la = len(arrays)
    dtype = np.result_type(*arrays)
    arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
    for i, a in enumerate(np.ix_(*arrays)):
        arr[...,i] = a
    return arr.reshape(-1, la)

def load_dataset(dataset, data_path):
    if dataset == 'movielens':
        data_path = os.path.join(data_path, 'ml-1m')
        df = load_movielens(data_path)
    
    elif dataset == 'netflix':
        data_path = os.path.join(data_path, 'netflix')
        df =  load_netflix(data_path, file_name='data.csv')

    elif dataset == 'pinterest':
        df = load_pinterest(data_path) 
    
    elif dataset == 'amazon_music':
        df = load_amazon_music(data_path)

    else:
        raise NotImplementedError
    if dataset == 'movielens':
        df.user_id -= 1
        df.item_id -= 1
    n_user = np.max(df.user_id) + 1
    n_item = np.max(df.item_id) + 1
    df.rating = (df.rating >= 4).astype(int)
    train, test = train_test_split(df, random_state=1000)
    df_all = pd.DataFrame(
        [[u, i] for u,i in product(range(n_user), range(n_item))],
        columns=["user_id", "item_id"]
    )
    
    df_all = pd.merge(
        df_all, 
        train[["user_id", "item_id", "rating"]], 
        on=["user_id", "item_id"], 
        how="left"
    )
    test = pd.merge(
        df_all[df_all.rating.isna()][["user_id", "item_id"]], 
        test[["user_id", "item_id", "rating"]], 
        on=["user_id", "item_id"], 
        how="left"
    ).fillna(0)
    train_set = train[train.rating == 1][["user_id", "item_id"]].values
    test_set = test[["user_id", "item_id", "rating"]].values
    return n_user, n_item, train_set, test_set
