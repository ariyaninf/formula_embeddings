import logging
import torch
import numpy as np
import pandas as pd
from pathlib import Path
import os.path
from torch.utils.data import TensorDataset, SequentialSampler, DataLoader

# pre_model = "bert-base-uncased"
# tokenizer = BertTokenizer.from_pretrained(pre_model, do_lower_case=True, cache_dir="./model/")


def init_logging_path(dir_log):
    dir_log = os.path.join(dir_log, f"log/")
    if os.path.exists(dir_log) and os.listdir(dir_log):
        dir_log += f'log_{len(os.listdir(dir_log)) + 1}.log'
        with open(dir_log, 'w'):
            os.utime(dir_log, None)
    if not os.path.exists(dir_log):
        os.makedirs(dir_log)
        dir_log += f'log_{len(os.listdir(dir_log)) + 1}.log'
        with open(dir_log, 'w'):
            os.utime(dir_log, None)
    return dir_log


def load_csv(fname):
    df = pd.read_csv(fname, sep=None, engine="python")
    df.drop(['k_sat', 'n_var', 'num_clauses'], axis=1, inplace=True)
    print('Number of pairs: {:,}\n'.format(df.shape[0]))
    return df


def load_pairs(x_train, x_val, x_test, file_loader_path, df_allpairs, batch_size):
    fname_train = file_loader_path + "_train.csv"
    fname_val = file_loader_path + "_val.csv"
    fname_test = file_loader_path + "_test.csv"
    if os.path.exists(fname_train):
        df_train = pd.read_csv(fname_train, sep=';', engine="python")
        df_val = pd.read_csv(fname_val, sep=';', engine="python")
        df_test = pd.read_csv(fname_test, sep=';', engine="python")
    else:
        df_train = pair_loader(x_train, file_loader_path, df_allpairs, batch_size, 'train')
        df_val = pair_loader(x_val, file_loader_path, df_allpairs, batch_size, 'val')
        df_test = pair_loader(x_test, file_loader_path, df_allpairs, batch_size, 'test')
    return df_train, df_val, df_test


def load_pairs_stack(x_train, x_val, file_loader_path, df_allpairs, batch_size):
    fname_train = file_loader_path + "_train.csv"
    fname_val = file_loader_path + "_val.csv"
    print(x_train.columns.values)
    print(x_val.columns.values)
    if os.path.exists(fname_train):
        df_train = pd.read_csv(fname_train, sep=';', engine="python")
        df_val = pd.read_csv(fname_val, sep=';', engine="python")
    else:
        df_train = pair_loader_stack(x_train, file_loader_path, df_allpairs, batch_size, 'train')
        df_val = pair_loader_stack(x_val, file_loader_path, df_allpairs, batch_size, 'val')
    return df_train, df_val


def pair_loader(dataset, file_loader_path, df_allpairs, batch_size, mode):
    fname_out = file_loader_path + "_" + mode + ".csv"
    # equ_size = int(batch_size / 4)
    pos_size = int(batch_size / 2)
    neg_size = int(batch_size / 2)

    pd_subset = pd.DataFrame(columns=['id', 'id_set', 'is_equivalent', 'is_entailed'])

    for index, row in dataset.iterrows():
        id_set = row["id_set"]

        '''
        # Get equivalent sentences
        df_equ = df_allpairs.loc[(df_allpairs['id_set'] == id_set) & (df_allpairs['is_equivalent'] == 1) & (df_allpairs['is_entailed'] == 1)]
        df_equ = df_equ[:equ_size]
        df_equ = df_equ.drop(columns=['sentence1', 'sentence2'])
        '''
        # Get positive entailments
        df_pos = df_allpairs.loc[(df_allpairs['id_set'] == id_set) & (df_allpairs['is_entailed'] == 1)]
        df_pos = df_pos[:pos_size]
        df_pos = df_pos.drop(columns=['sentence1', 'sentence2'])
        # Get negative entailments
        df_neg = df_allpairs.loc[(df_allpairs['id_set'] == id_set) & (df_allpairs['is_entailed'] == 0)]
        df_neg = df_neg[:neg_size]
        df_neg = df_neg.drop(columns=['sentence1', 'sentence2'])

        # pd_subset = pd.concat([pd_subset, df_equ]).reset_index(drop=True)
        pd_subset = pd.concat([pd_subset, df_pos]).reset_index(drop=True)
        pd_subset = pd.concat([pd_subset, df_neg]).reset_index(drop=True)

    pd_subset.to_csv(fname_out, sep=';', encoding='utf-8', index=False)
    print(len(pd_subset), ' rows created.')
    return pd_subset


def pair_loader_stack(dataset, file_loader_path, df_allpairs, batch_size, mode):
    fname_out = file_loader_path + "_" + mode + ".csv"
    # equ_size = int(batch_size / 4)
    pos_size = int(batch_size / 2)
    neg_size = int(batch_size / 2)

    pd_subset = pd.DataFrame(columns=['id', 'id_set', 'is_equivalent', 'is_entailed'])

    for index, row in dataset.iterrows():
        id_set = row["id_set"]

        # Get positive entailments
        df_pos = df_allpairs.loc[(df_allpairs['id_set'] == id_set) & (df_allpairs['is_entailed'] == 1)]
        df_pos = df_pos[:pos_size]
        df_pos = df_pos.drop(columns=['sentence1', 'sentence2'])
        print('df_pos')
        print(df_pos)
        # Get negative entailments
        df_neg = df_allpairs.loc[(df_allpairs['id_set'] == id_set) & (df_allpairs['is_entailed'] == 0)]
        df_neg = df_neg[:neg_size]
        df_neg = df_neg.drop(columns=['sentence1', 'sentence2'])

        # pd_subset = pd.concat([pd_subset, df_equ]).reset_index(drop=True)
        pd_subset = pd_subset.append(df_pos,
                                     ignore_index=True)
        # pd_subset = pd.concat([pd_subset, df_pos], axis=0)
        print('pd_subset')
        print(pd_subset)
        pd_subset = pd.concat([pd_subset, df_neg]).reset_index(drop=True)

    print(pd_subset)

    pd_subset.to_csv(fname_out, sep=';', encoding='utf-8-sig', index=False)
    print(len(pd_subset), ' rows created.')
    return pd_subset


def dataloader(df_pairs, batch_size):
    df_pairs = np.array(df_pairs, dtype=np.float32)
    df_pairs_tensor = torch.tensor(df_pairs, dtype=torch.float32)
    dataset = TensorDataset(df_pairs_tensor)
    sampler = SequentialSampler(dataset)
    loader = DataLoader(dataset=dataset, batch_size=batch_size, sampler=sampler)
    return loader


def dataloader_rplp(df_pairs, batch_size):
    df_pairs = df_pairs.drop(columns=['facts', 'rules', 'queries', 'num_preds', 'num_rules', 'num_facts'])
    df_pairs = np.array(df_pairs, dtype=np.float32)
    df_pairs_tensor = torch.tensor(df_pairs, dtype=torch.float32)
    dataset = TensorDataset(df_pairs_tensor)
    sampler = SequentialSampler(dataset)
    loader = DataLoader(dataset=dataset, batch_size=batch_size, sampler=sampler)
    return loader

