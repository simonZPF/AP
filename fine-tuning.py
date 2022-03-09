import json
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import *

from DAO import FewShotSet, CSVDataSet
from config import *

model = torch.load("temp/temp_eng_demo5_5")
# model = BertForMaskedLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
classes_id = [[tokenizer.convert_tokens_to_ids(i) for i in word] for word in classes_word]
classes_id = torch.Tensor(classes_id).long().T
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate(model, data_iter):
    with torch.no_grad():
        model.eval()
        model.to(device)
        y_preds = torch.Tensor()
        y_true = torch.Tensor()
        for x, y in data_iter:
            token, segment = x
            probas = model(token.to(device), segment.to(device))[0]
            probas = probas.cpu()
            # print(x[0][0])
            y_pred = torch.zeros(len(y), len(classes_word))
            for i, m in enumerate(mask_ids):
                y_pred += probas[:, m + 1, classes_id[i]]
            y_true = torch.cat((y_true, y))
            y_preds = torch.cat((y_preds, y_pred.argmax(axis=1)))

        # print(confusion_matrix(y_true, y_preds))
        # print(accuracy_score(y_true, y_preds))
        return accuracy_score(y_true, y_preds)


def train_classify(net, train_iter, device, optim, loss, epochs=5, dev_iter=None, test_iter=None):
    net = net.to(device)
    print("training on ", device)
    test_acc = []
    dev_acc = []
    train_acc = []
    train_loss = []
    for i in range(epochs):
        net.train()
        batch_count, train_l, start = 0, 0.0, time.time()
        y_true = torch.Tensor()
        y_preds = torch.Tensor()
        for x, y in tqdm(train_iter):
            token, segment = x
            probas = net(token.to(device), segment.to(device))[0]
            # y_pred = torch.zeros(len(y), len(classes_word)).cuda()
            # for i, m in enumerate(mask_ids):
            #     y_pred += probas[:, m + 1, classes_id[i]]
            y_pred = probas[:, 1, classes_id[0]]
            y = y.cuda()
            l = loss(y_pred, y)
            optim.zero_grad()
            l.backward()
            optim.step()
            train_l += l.cpu().item()
            y_preds = torch.cat((y_preds, y_pred.cpu().argmax(axis=1)))
            y_true = torch.cat((y_true, y.cpu()))
            batch_count += 1
        # print(confusion_matrix(y_true, y_preds))
        train_l /= batch_count
        tc = accuracy_score(y_true, y_preds)
        print(" %d loss: %.5f train acc: %.5f" % (i + 1, train_l, tc))
        train_loss.append(train_l)
        train_acc.append(tc)
        if dev_iter is not None:
            mean_dc = []
            for k, dev in enumerate(dev_iter):
                dc = evaluate(model, dev)
                print(f"dev{k} acc %.5f " % dc, end=" ")
                mean_dc.append(dc)
            dev_acc.append(mean_dc)
        if test_iter is not None:
            t = evaluate(model, test_iter)
            print("test acc %.5f " % t)
            test_acc.append(t)
        time.sleep(0.1)
        if i % 5 == 0:
            torch.save(model, f"temp/temp_{i}")
    return train_acc, train_loss, dev_acc, test_acc


for name, p in model.named_parameters():
    p.requires_grad = True

optim = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=2e-6)
loss = nn.CrossEntropyLoss()
# train_data = FewShotSet("data/few_shot/dev_0.json")
# train_iter = DataLoader(train_data, batch_size=5, shuffle=True)
# test_data = FewShotSet("data/few_shot/test_public.json")
# test_iter = DataLoader(test_data, batch_size=5, shuffle=True)
data = CSVDataSet("data/eng/sst/train.csv", rate=1)
train, test = train_test_split(data, test_size=0.8, random_state=42)
train_iter = DataLoader(train[:32], batch_size=16, shuffle=True)
test_iter = DataLoader(test[:600], batch_size=16, shuffle=True)

train_acc, train_loss, dev_acc, test_acc = \
    train_classify(model, train_iter, device, optim, loss, epochs=30,
                   test_iter=test_iter)
