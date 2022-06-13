import matplotlib.pyplot as plt
import glob
import os
import time
import numpy as np
import torch
import torch.nn.functional as F
from models import Model
from torch.utils.data import random_split
from torch_geometric.data import DataLoader
from torch_geometric.datasets import TUDataset
import torch
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from torch_geometric.nn import GCNConv

from layers import GCN, HGPSLPool
from mtm_1_reg import StaticGraphTemporalSignal
import argparse

#
# dataset = TUDataset(os.path.join('data', 'ENZYMES'), name='ENZYMES', use_node_attr=True)
# print(dataset[0].x.shape)
# train_loader = DataLoader(dataset, batch_size=512)
# for i, data in enumerate(train_loader):
#     print(data.x.shape)

from mtm_1_reg import MTMDatasetLoader

loader = MTMDatasetLoader()

dataset = loader.get_dataset()
# print(dataset.targets)
# list=[]
# for i in range(len(dataset.targets)):
#     list.append(dataset.targets[i][0])
# print(list)

# pre_model = "./84.pth"
# dict = torch.load(pre_model)
# for key in list(dict.keys()):
#     # print(key)
#     if key.startswith('lin'):
#         del dict[key]
# torch.save(dict, './model_deleted.pth')
# # # #验证修改是否成功
# changed_dict = torch.load('./model_deleted.pth')
# for key in dict.keys():
#     print(key)
parser = argparse.ArgumentParser()

parser.add_argument('--seed', type=int, default=777, help='random seed')
parser.add_argument('--batch_size', type=int, default=512, help='batch size')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=0.001, help='weight decay')
parser.add_argument('--nhid', type=int, default=128, help='hidden size')
parser.add_argument('--sample_neighbor', type=bool, default=True, help='whether sample neighbors')
parser.add_argument('--sparse_attention', type=bool, default=True, help='whether use sparse attention')
parser.add_argument('--structure_learning', type=bool, default=True, help='whether perform structure learning')
parser.add_argument('--pooling_ratio', type=float, default=0.8, help='pooling ratio')
parser.add_argument('--dropout_ratio', type=float, default=0.0, help='dropout ratio')
parser.add_argument('--lamb', type=float, default=1.0, help='trade-off parameter')
parser.add_argument('--dataset', type=str, default='MTM_1', help='DD/PROTEINS/NCI1/NCI109/Mutagenicity/ENZYMES')
parser.add_argument('--device', type=str, default='cpu', help='specify cuda devices')
parser.add_argument('--epochs', type=int, default=100, help='maximum number of epochs')
parser.add_argument('--patience', type=int, default=100, help='patience for early stopping')
args = parser.parse_args()
args.num_classes = 6
args.num_features = 3

torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)


class MyModel(torch.nn.Module):
    def __init__(self, args):
        super(MyModel, self).__init__()
        self.args = args
        self.num_features = args.num_features
        self.nhid = args.nhid
        self.num_classes = args.num_classes
        self.pooling_ratio = args.pooling_ratio
        self.dropout_ratio = args.dropout_ratio
        self.sample = args.sample_neighbor
        self.sparse = args.sparse_attention
        self.sl = args.structure_learning
        self.lamb = args.lamb

        self.conv1 = GCNConv(self.num_features, self.nhid)
        self.conv2 = GCN(self.nhid, self.nhid)
        self.conv3 = GCN(self.nhid, self.nhid)

        self.pool1 = HGPSLPool(self.nhid, self.pooling_ratio, self.sample, self.sparse, self.sl, self.lamb)
        self.pool2 = HGPSLPool(self.nhid, self.pooling_ratio, self.sample, self.sparse, self.sl, self.lamb)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        edge_attr = None

        x = F.relu(self.conv1(x, edge_index, edge_attr))
        x, edge_index, edge_attr, batch = self.pool1(x, edge_index, edge_attr, batch)
        x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.conv2(x, edge_index, edge_attr))
        x, edge_index, edge_attr, batch = self.pool2(x, edge_index, edge_attr, batch)
        x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.conv3(x, edge_index, edge_attr))
        x3 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(x1) + F.relu(x2) + F.relu(x3)

        return x

def dataset_split(dataset,train_ratio):
    train_snapshots = int(train_ratio * dataset.snapshot_count)
    train_snapshots = StaticGraphTemporalSignal(
            dataset.edge_index,
            dataset.edge_weight,
            dataset.features[0:train_snapshots],
            dataset.targets[0:train_snapshots],
            **{key: getattr(dataset, key)[0:train_snapshots] for key in dataset.additional_feature_keys}
        )
    return train_snapshots

mymodel=MyModel(args).to(args.device)
mymodel.load_state_dict(torch.load('model_deleted.pth'))
xlist=[]
ylist=[]
dataset=dataset_split(dataset,0.1)
# print(dataset[0].x.detach().numpy())
data=dataset[1000].to(args.device)
x=mymodel(data)
print(len(x.detach().numpy()[0]))
# for i,data in enumerate(dataset):
#     data=data.to(args.device)
#     x=mymodel(data)
#     numpyx=x.detach().numpy()[0]
#     list=[]
#     for xx in numpyx:
#         list.append(float(xx))
#     xlist.append(list)
    # ylist.append(data.y.numpy()[0])

# with open("./x.txt","w") as f:
#     f.write(str(xlist))
# with open("./y.txt","w") as f:
#     f.write(str(ylist))
# with open("./y.txt","r") as f:
#     y=f.read()