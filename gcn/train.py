import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim
from model import GCN
from utils import load_data

model = GCN(1433, 16, 7)
print(model)

# #load dataset

# def load_cora_data():
#     dataset = CoraGraphDataset()
#     g = dataset[0]
#     features = g.ndata['feat']
#     labels = g.ndata['label']
#     train_mask = g.ndata['train_mask']
#     test_mask = g.ndata['test_mask']
#     return g, features, labels, train_mask, test_mask


# g, features, labels, train_mask, test_mask = load_cora_data()

#load data

adj, features, labels, idx_train, idx_val, idx_test = load_data()
# features = features.view(1,features.shape[0], features.shape[1])
# adj = adj.view(1,adj.shape[0],adj.shape[1])

#feature shape
#torch.Size([2708, 1433])
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def train():
    model.train()
    optimizer.zero_grad()
    output = model(features, adj)
    loss_train = F.nll_loss(output[train_mask], labels[test_mask])
    loss_train.backward()
    optimizer.step()
    
for epoch in range(1, 10):
    print("epoch:", epoch)
    train()
    
    
    





