import json

import pandas as pd
import torch
from transformers import AutoTokenizer
from config import *
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from sklearn.model_selection import train_test_split

tokenizer = AutoTokenizer.from_pretrained(model_name)

MASK = '[MASK]'
CLS = '[CLS]'
SEP = '[SEP]'
PAD = '[PAD]'
if "roberta-" in model_name:
    MASK = '<mask>'
    CLS = '<s>'
    SEP = '</s>'
    PAD = '<pad>'


def mask_tokenize(mask_ids, string, pre_str="", post_str=""):
    tokenized_str = tokenizer.tokenize(string)
    tl = len(tokenized_str)
    tokenized_str = tokenizer.tokenize(pre_str) + tokenized_str[:MAX_LEN] + tokenizer.tokenize(post_str)
    for i in mask_ids:
        tokenized_str[i] = MASK
    tokenized_str = [CLS] + tokenized_str + [SEP]
    if tl < MAX_LEN:
        tokenized_str += max(0, MAX_LEN - tl) * [PAD]
    token_id = tokenizer.convert_tokens_to_ids(tokenized_str)
    segment = [0] * len(token_id)
    return torch.tensor(token_id).long(), torch.tensor(segment).long()


class CSVDataSet(Dataset):
    def __init__(self, file, rate=1.0, pre_str=pre_str):
        self.file = file
        self.rate = rate
        self.pre_str = pre_str
        self.X, self.Y = self.get_data()

    def get_data(self):
        try:
            df = pd.read_csv(self.file)
        except Exception as e:
            df = pd.read_csv(self.file, encoding="utf-8")
        df = df.dropna(subset=["review"])
        df = df[df["label"].isin([0, 1])]
        if 'cat' in df.columns.values:
            df = df[~df['cat'].isin(['计算机', '平板', '手机'])]
        data = df.sample(frac=self.rate)
        data["review"] = data["review"].apply(lambda x: mask_tokenize(mask_ids, x, pre_str=self.pre_str, post_str=""))

        return data["review"], data["label"]

    def __getitem__(self, item):
        return self.X.iloc[item], self.Y.iloc[item]

    def __len__(self):
        return len(self.X)


class FewShotSet(Dataset):
    def __init__(self, filename, pre_str=pre_str):
        self.filename = filename
        self.pre_str = pre_str
        self.X, self.Y = self.get_data()

    def get_data(self):
        label2int = {'Negative': 0, 'Positive': 1}
        X = []
        Y = []
        with open(self.filename, encoding='utf-8') as f:
            for line in f.readlines():
                data = json.loads(line)
                X.append(mask_tokenize(mask_ids, data['sentence'], pre_str=self.pre_str, ))
                Y.append(label2int[data['label']])

        return X, Y

    def __getitem__(self, item):
        return self.X[item], self.Y[item]

    def __len__(self):
        return len(self.X)


# dataset_list = [
#     {"name": "hotel", "path": "data/hotel.csv", "rate": rate},  # 7765
#     {"name": "weibo", "path": "data/weibo_clean.csv", "rate": rate},  # 119988
#     {"name": "shopping", "path": "data/online_shopping_10_cats.csv", "rate": rate},  # 62773
#     {"name": "waimai", "path": "data/waimai_10k.csv", "rate": rate},  # 11987
# ]
# dataset_list = [
#     {"name": "amazon", "path": "data/eng/amazon/amazon.csv", "rate": rate},  # 3600000
#     {"name": "twitter", "path": "data/eng/twitter/twitter.csv", "rate": rate * 3},
#     {"name": "finance", "path": "data/eng/finance/finance.csv", "rate": 1},
# ]
# #
# train_data = []
# dev_data = []
# dev_iter = []
# for dataset in dataset_list:
#     data = CSVDataSet(dataset['path'], rate=dataset['rate'])
#     train, dev = train_test_split(data, test_size=0.2, random_state=42)
#     train_data.append(train)
#     dev_data.append(dev)
# mix_data = ConcatDataset(train_data)
# train_iter = DataLoader(mix_data, batch_size=batch_size, shuffle=True)
# for dev in dev_data:
#     dev_iter.append(DataLoader(dev, batch_size=batch_size, shuffle=True))
#
# test_data = CSVDataSet("data/eng/sst/train.csv", rate=0.1)
# test_iter = DataLoader(test_data, batch_size=batch_size, shuffle=True)

# data = CSVDataSet("data/film_train.csv", rate=1)
# train, dev = train_test_split(data, test_size=0.2, random_state=42)
# print(train[0])
# s=[]
# for i in range(4):
#     df = pd.read_csv(dataset_list[i]['path'], encoding="utf-8")
#     df = df.dropna(subset=["review"])
#     s.append(df)
# d = pd.concat(s)
# print(d)
# d['len'] = d["review"].apply(lambda x:len(x))
# print(d['len'].describe())
# ""
# "56 44"

# data = CSVDataSet("data/eng/sst/train.csv", rate=0.01)
#
# print(data[0])

# good = tokenizer.tokenize("it is good")
# print(good)
# bad = tokenizer.tokenize("bad")
# print(bad)
# "2213, 1363"
# print(tokenizer.convert_tokens_to_ids(good))
# print(tokenizer.convert_tokens_to_ids(bad))
# print(tokenizer.convert_tokens_to_ids(CLS))
# print(tokenizer.convert_tokens_to_ids(PAD))
