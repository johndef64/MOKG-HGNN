# GCN_run_pyg.py
# Verione di GCN_run.py adattata per PyTorch Geometric, usa edge_index invece di adj_hat
# e usa Data di PyG per gestire i dati

#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/8/8 16:43
# @Author  : Li Xiao
# @File    : GCN_run_pyg.py (adapted for PyTorch Geometric)

import numpy as np
import pandas as pd
import argparse
import glob
import os
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from GCN_pyg import GCN
from utils import load_data, metrics_calc


def setup_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

def accuracy(output, labels):
    preds = output.max(1)[1]
    return preds.eq(labels).sum().item() / len(labels)

def train(model, optimizer, data, idx_train):
    model.train()
    optimizer.zero_grad()
    output = model(data.x, data.edge_index)
    loss = F.cross_entropy(output[idx_train], data.y[idx_train])
    loss.backward()
    optimizer.step()
    return loss.item()

def test(model, data, idx_test):
    model.eval()
    output = model(data.x, data.edge_index)
    loss = F.cross_entropy(output[idx_test], data.y[idx_test]).item()
    preds = output[idx_test].max(1)[1].cpu().numpy()
    labels = data.y[idx_test].cpu().numpy()
    acc = (preds == labels).sum() / len(labels)
    f1 = f1_score(labels, preds, average='weighted')
    print("Test set results:", f"loss= {loss:.4f}", f"accuracy= {acc:.4f}")
    return acc, f1

def predict(model, data, sample_names, idx):
    model.eval()
    output = model(data.x, data.edge_index)
    pred = output.detach().cpu().numpy()
    pred = np.argmax(pred, axis=1).tolist()
    df = pd.DataFrame({'Sample': sample_names, 'predict_label': pred})
    df = df.iloc[idx.tolist(), :]
    df.to_csv('result/GCN_pyg_predicted_data.csv', header=True, index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--featuredata', '-fd', type=str, required=True)
    parser.add_argument('--adjdata', '-ad', type=str, required=True)
    parser.add_argument('--labeldata', '-ld', type=str, required=True)
    parser.add_argument('--testsample', '-ts', type=str)
    parser.add_argument('--mode', '-m', type=int, choices=[0,1], default=0)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--device', '-d', type=str, choices=['cpu', 'gpu'], default='cpu')
    parser.add_argument('--epochs', '-e', type=int, default=150)
    parser.add_argument('--learningrate', '-lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', '-w', type=float, default=0.01)
    parser.add_argument('--hidden', '-hd',type=int, default=64)
    parser.add_argument('--dropout', '-dp', type=float, default=0.5)
    parser.add_argument('--threshold', '-t', type=float, default=0.005)
    parser.add_argument('--nclass', '-nc', type=int, default=4)
    parser.add_argument('--patience', '-p', type=int, default=20)
    args = parser.parse_args()

    device = torch.device('cuda' if args.device == 'gpu' and torch.cuda.is_available() else 'cpu')
    setup_seed(args.seed)

    # Usa PyG: edge_index invece di adj_hat
    edge_index, features, labels, sample_names = load_data(args.adjdata, args.featuredata, args.labeldata, mode=1, threshold=args.threshold)
    data = Data(x=features, edge_index=edge_index, y=labels).to(device)

    print('Begin training model...')

    if args.mode == 0:
        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=args.seed)
        acc_res, f1_res = [], []

        for train_idx, test_idx in skf.split(features.cpu(), labels.cpu()):
            model = GCN(n_in=features.shape[1], n_hid=args.hidden, n_out=args.nclass, dropout=args.dropout).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=args.learningrate, weight_decay=args.weight_decay)

            train_idx = torch.tensor(train_idx, dtype=torch.long, device=device)
            test_idx = torch.tensor(test_idx, dtype=torch.long, device=device)

            for epoch in range(args.epochs):
                loss = train(model, optimizer, data, train_idx)
                if (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch+1} | Loss: {loss:.4f}")

            acc, f1 = test(model, data, test_idx)
            acc_res.append(acc)
            f1_res.append(f1)

        print('10-fold  Acc(%.4f, %.4f)  F1(%.4f, %.4f)' % (
            np.mean(acc_res), np.std(acc_res), np.mean(f1_res), np.std(f1_res)))

    elif args.mode == 1:
        test_sample_df = pd.read_csv(args.testsample)
        test_sample = test_sample_df.iloc[:, 0].tolist()
        all_sample = sample_names
        train_sample = list(set(all_sample) - set(test_sample))

        train_idx = [i for i, name in enumerate(all_sample) if name in train_sample]
        test_idx = [i for i, name in enumerate(all_sample) if name in test_sample]

        model = GCN(n_in=features.shape[1], n_hid=args.hidden, n_out=args.nclass, dropout=args.dropout).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learningrate, weight_decay=args.weight_decay)

        train_idx = torch.tensor(train_idx, dtype=torch.long, device=device)
        test_idx = torch.tensor(test_idx, dtype=torch.long, device=device)

        loss_values = []
        bad_counter, best_epoch = 0, 0
        best = float('inf')

        for epoch in range(args.epochs):
            loss = train(model, optimizer, data, train_idx)
            loss_values.append(loss)

            if loss < best:
                best = loss
                best_epoch = epoch
                bad_counter = 0
            else:
                bad_counter += 1
            if not os.path.exists('model/GCN_pyg'):
                os.makedirs('model/GCN_pyg')
            torch.save(model.state_dict(), f'model/GCN_pyg/{epoch}.pkl')

            if bad_counter == args.patience:
                break

            for f in glob.glob('model/GCN_pyg/*.pkl'):
                ep = int(os.path.basename(f).split('.')[0])
                if ep != best_epoch:
                    os.remove(f)

        print('Training finished. Best epoch:', best_epoch)
        model.load_state_dict(torch.load(f'model/GCN_pyg/{best_epoch}.pkl'))
        predict(model, data, all_sample, test_idx)

        if os.path.exists(f'result/GCN_pyg_predicted_data.csv'):
            acc, f1 = metrics_calc('data/sample_classes.csv', 'result/GCN_pyg_predicted_data.csv')
            print(f'Prediction accuracy on test set: {acc:.4f}, F1-score (weighted): {f1:.4f}')

    print('Finished!')