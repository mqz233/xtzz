import argparse
import matplotlib.pyplot as plt
import glob
import os
import time
import numpy
import torch
import torch.nn.functional as F
from models import Model
from torch.utils.data import random_split
from torch_geometric.data import DataLoader
from torch_geometric.datasets import TUDataset
from mtm_1_reg import MTMDatasetLoader,StaticGraphTemporalSignal
from torch_geometric_temporal.signal import temporal_signal_split


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
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)

# dataset = TUDataset(os.path.join('data', args.dataset), name=args.dataset, use_node_attr=True)


# num_training = int(len(dataset) * 0.8)
# num_val = int(len(dataset) * 0.1)
# num_test = len(dataset) - (num_training + num_val)
# training_set, validation_set, test_set = random_split(dataset, [num_training, num_val, num_test])
#
# train_loader = DataLoader(training_set, batch_size=args.batch_size, shuffle=True)
# val_loader = DataLoader(validation_set, batch_size=args.batch_size, shuffle=False)
# test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

# whole_loader = DataLoader(dataset)  ###


def train():
    min_loss = 1e10
    patience_cnt = 0
    val_loss_values = []
    best_epoch = 0

    t = time.time()
    model.train()

    for epoch in range(args.epochs):
        loss_train = 0.0
        correct = 0
        for i, data in enumerate(train_loader):
            optimizer.zero_grad()
            data = data.to(args.device)
            out = model(data)
            loss = F.nll_loss(out, data.y)
            loss.backward()
            optimizer.step()
            loss_train += loss.item()
            pred = out.max(dim=1)[1]
            correct += pred.eq(data.y).sum().item()
        acc_train = correct / len(train_loader.dataset)
        acc_val, loss_val = compute_test(val_loader)
        print('Epoch: {:04d}'.format(epoch + 1), 'loss_train: {:.6f}'.format(loss_train),
              'acc_train: {:.6f}'.format(acc_train), 'loss_val: {:.6f}'.format(loss_val),
              'acc_val: {:.6f}'.format(acc_val), 'time: {:.6f}s'.format(time.time() - t))

        train_loss.append(loss_train)
        valid_loss.append(loss_val)

        val_loss_values.append(loss_val)
        torch.save(model.state_dict(), '{}.pth'.format(epoch))
        if val_loss_values[-1] < min_loss:
            min_loss = val_loss_values[-1]
            best_epoch = epoch
            patience_cnt = 0
        else:
            patience_cnt += 1

        # if patience_cnt == self.patience:
        #     break

        files = glob.glob('*.pth')
        for f in files:
            epoch_nb = int(f.split('.')[0])
            if epoch_nb < best_epoch:
                os.remove(f)

    files = glob.glob('*.pth')
    for f in files:
        epoch_nb = int(f.split('.')[0])
        if epoch_nb > best_epoch:
            os.remove(f)
    print('Optimization Finished! Total time elapsed: {:.6f}'.format(time.time() - t))

    return best_epoch


def compute_test(loader):
    model.eval()
    correct = 0.0
    loss_test = 0.0

    for data in loader:
        data = data.to(args.device)
        out = model(data)
        pred = out.max(dim=1)[1]
        correct += pred.eq(data.y).sum().item()
        loss_test += F.nll_loss(out, data.y).item()

        # print(pred.numpy().tolist())  ###
        # print(data.y.numpy().tolist())

    return correct / len(loader.dataset), loss_test


def compute_all(loader, pre:list, real:list):
    model.eval()
    correct = 0.0
    loss_test = 0.0

    for data in loader:
        data = data.to(args.device)
        out = model(data)
        pred = out.max(dim=1)[1]
        correct += pred.eq(data.y).sum().item()
        loss_test += F.nll_loss(out, data.y).item()

        pre.append(pred.item())
        real.append(data.y.item())

    return correct / len(loader.dataset), loss_test

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


if __name__ == '__main__':
    loader = MTMDatasetLoader()
    dataset = loader.get_dataset()
    dataset = dataset_split(dataset, train_ratio=0.05)
    num_training = int(len(dataset) * 0.8)
    num_val = int(len(dataset) * 0.1)
    num_test = len(dataset) - (num_training + num_val)
    training_set, validation_set, test_set = random_split(dataset, [num_training, num_val, num_test])

    train_loader = DataLoader(training_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(validation_set, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)
    whole_loader = DataLoader(dataset)
    args.num_classes = 6
    args.num_features = 3

    print(args)

    # train_dataset, test_dataset = temporal_signal_split(dataset, train_ratio=0.2)

    model = Model(args).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    train_loss = []
    valid_loss = []
    # Model training
    best_model = train()
    # Restore best model for test set
    model.load_state_dict(torch.load('{}.pth'.format(best_model)))
    # test_acc, test_loss = compute_test(test_loader)
    plt.figure()
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.plot(train_loss, 'r', label='train_loss')
    plt.plot(valid_loss, 'b', label='valid_loss')
    plt.legend(loc='best')
    plt.savefig('./mtm_loss.png')

    pre = []
    real = []
    test_acc, test_loss = compute_all(whole_loader,pre,real)
    print('Test set results, loss = {:.6f}, accuracy = {:.6f}'.format(test_loss, test_acc))
    print(pre)
    print(real)
    # pre = pre[::10]
    # real = real[::10]
    plt.figure()
    plt.xlabel("number")  # x轴上的名字
    plt.ylabel("label")  # y轴上的名字
    # x = []
    # for i in range(600):
    #     x.append(i)
    # x = x[::10]

    plt.plot(pre, 'r', label='prediction')
    plt.plot(real, 'b', label='real')

    # plt.plot(x, pre, 'r', label='prediction')
    # plt.plot(x, real, 'b', label='real')
    #
    # # plt.scatter(x,pre,color='red',label='prediction')
    # # plt.scatter(x,real,color='blue',label='real')
    #
    plt.legend(loc='best')
    plt.savefig('./mtm_label.png')
