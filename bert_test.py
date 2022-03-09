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

from DAO import CSVDataSet, train_iter, dev_iter, test_iter, dataset_list
from config import *
from lstm_att_bert import GRU_ATT_PET
from lstm_bert import LSTM_PET

if model_type == "PET":
    model = BertForMaskedLM.from_pretrained(model_name)
else:
    model = GRU_ATT_PET(model_name, steps=steps)
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
        string = " %d loss: %.5f train acc: %.5f " % (i + 1, train_l, tc)
        if dev_iter is not None:
            mean_dc = []
            for k, dev in enumerate(dev_iter):
                dc = evaluate(model, dev)
                print(f"dev{k} acc %.5f " % dc, end=" ")
                string += f"dev{k} acc %.5f " % dc
                mean_dc.append(dc)
            string += f"mean_dev acc %.5f " % np.mean(mean_dc)
            dev_acc.append(mean_dc)

        if test_iter is not None:
            t = evaluate(model, test_iter)
            print("test acc %.5f " % t)
            string += f"test acc %.5f \n" % t
            test_acc.append(t)
        time.sleep(0.1)
        with open(f"log/temp_{test_name}.txt", "a") as f:
            f.write(string)
        if i % 5 == 0:
            torch.save(model, f"temp/temp_{test_name}_{i}")
    return train_acc, train_loss, dev_acc, test_acc


for name, p in model.named_parameters():
    if "bert" in name:
        p.requires_grad = False

optim = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
loss = nn.CrossEntropyLoss()

# data = CSVDataSet("data/eng/sst/train.csv", rate=1)
# train, test = train_test_split(data, test_size=0.8, random_state=42)
# train_iter = DataLoader(train[:32], batch_size=16, shuffle=True)
# test_iter = DataLoader(test[:600], batch_size=24, shuffle=True)
#
train_acc, train_loss, dev_acc, test_acc = \
    train_classify(model, train_iter, device, optim, loss, epochs=epochs,
                   dev_iter=dev_iter, test_iter=test_iter)

plt.plot(train_acc, label="train_acc")
if dev_acc:
    plt.plot([np.mean(acc) for acc in dev_acc], label="dev_acc")
if test_acc:
    plt.plot(test_acc, label="test_acc")
plt.legend(loc='lower right')
plt.savefig(f"img/{test_name}_1.png", dpi=300)
plt.clf()
if dev_acc:
    plt.plot(train_acc, label="train_acc")
    for i, dataset in enumerate(dataset_list):
        plt.plot([acc[i] for acc in dev_acc], label=dataset['name'])
plt.legend(loc='upper right')
plt.savefig(f"img/{test_name}_2.png", dpi=300)
with open(f"log/{test_name}.txt", "w") as f:
    string = ""
    for i, j, k, l in zip(train_acc, train_loss, dev_acc, test_acc):
        string += "loss: %.5f  train_acc: %.5f dev_acc %.5f test_acc %.5f\n" % (j, i, np.mean(k), l)
    f.write(string)
torch.save(model, f"model/{test_name}")
