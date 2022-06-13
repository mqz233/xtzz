import matplotlib.pyplot as plt
import numpy as np
import glob
import os
import time
import numpy
import torch
import torch.nn.functional as F
from HGP.models import Model
from torch.utils.data import random_split
from torch_geometric.data import DataLoader
from torch_geometric.datasets import TUDataset
from HGP.mtm_1_reg import MTMDatasetLoader, StaticGraphTemporalSignal
from HGP.pos_reg2 import MyDatasetLoader


class HGP:
    def __init__(self, measurement, seed=777, batch_size=512, lr=0.001, weight_decay=0.001, nhid=128,
                 sample_neighbor=True,
                 sparse_attention=True,
                 structure_learning=True, pooling_ratio=0.8, dropout_ratio=0.0, lamb=1.0, device='cpu', epochs=300,
                 patience=100):
        self.measurenment = measurement
        self.dataset = 'ENZYMES'
        self.seed = seed
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.nhid = nhid
        self.sample_neighbor = sample_neighbor
        self.sparse_attention = sparse_attention
        self.structure_learning = structure_learning
        self.pooling_ratio = pooling_ratio
        self.dropout_ratio = dropout_ratio
        self.lamb = lamb
        self.device = device
        self.epochs = epochs
        self.patience = patience

    def HGP(self):
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.seed)
        dataset = TUDataset(os.path.join('data', self.dataset), name=self.dataset, use_node_attr=True)
        self.num_classes = dataset.num_classes
        self.num_features = dataset.num_features

        # print(self)

        num_training = int(len(dataset) * 0.8)
        num_val = int(len(dataset) * 0.1)
        num_test = len(dataset) - (num_training + num_val)
        training_set, validation_set, test_set = random_split(dataset, [num_training, num_val, num_test])

        train_loader = DataLoader(training_set, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(validation_set, batch_size=self.batch_size, shuffle=False)
        test_loader = DataLoader(test_set, batch_size=self.batch_size, shuffle=False)

        whole_loader = DataLoader(dataset)  ###

        model = Model(self).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        train_loss = []
        valid_loss = []

        def train():
            min_loss = 1e10
            patience_cnt = 0
            val_loss_values = []
            best_epoch = 0

            t = time.time()
            model.train()

            for epoch in range(self.epochs):
                loss_train = 0.0
                correct = 0
                for i, data in enumerate(train_loader):
                    optimizer.zero_grad()
                    data = data.to(self.device)
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
                data = data.to(self.device)
                out = model(data)
                pred = out.max(dim=1)[1]
                correct += pred.eq(data.y).sum().item()
                loss_test += F.nll_loss(out, data.y).item()

                # print(pred.numpy().tolist())  ###
                # print(data.y.numpy().tolist())

            return correct / len(loader.dataset), loss_test

        def compute_all(loader, pre: list, real: list):
            model.eval()
            correct = 0.0
            loss_test = 0.0

            for data in loader:
                data = data.to(self.device)
                out = model(data)
                pred = out.max(dim=1)[1]
                correct += pred.eq(data.y).sum().item()
                loss_test += F.nll_loss(out, data.y).item()

                pre.append(pred.item())
                real.append(data.y.item())

            return correct / len(loader.dataset), loss_test

        # Model training
        best_model = train()
        # Restore best model for test set
        model.load_state_dict(torch.load('{}.pth'.format(best_model)))
        # test_acc, test_loss = compute_test(test_loader)

        pre = []
        real = []
        test_acc, test_loss = compute_all(whole_loader, pre, real)
        print('Test set results, loss = {:.6f}, accuracy = {:.6f}'.format(test_loss, test_acc))
        print(pre)
        print(real)
        return pre, real, train_loss, valid_loss

    def MTM(self):

        def dataset_split(dataset, train_ratio):
            train_snapshots = int(train_ratio * dataset.snapshot_count)
            train_snapshots = StaticGraphTemporalSignal(
                dataset.edge_index,
                dataset.edge_weight,
                dataset.features[0:train_snapshots],
                dataset.targets[0:train_snapshots],
                **{key: getattr(dataset, key)[0:train_snapshots] for key in dataset.additional_feature_keys}
            )
            return train_snapshots

        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.seed)

        loader = MTMDatasetLoader()
        dataset = loader.get_dataset()
        dataset = dataset_split(dataset, train_ratio=0.1)
        self.num_classes = 6
        self.num_features = 3

        num_training = int(len(dataset) * 0.8)
        num_val = int(len(dataset) * 0.1)
        num_test = len(dataset) - (num_training + num_val)
        training_set, validation_set, test_set = random_split(dataset, [num_training, num_val, num_test])

        train_loader = DataLoader(training_set, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(validation_set, batch_size=self.batch_size, shuffle=False)
        test_loader = DataLoader(test_set, batch_size=self.batch_size, shuffle=False)

        whole_loader = DataLoader(dataset)  ###

        model = Model(self).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        train_loss = []
        valid_loss = []

        def train():
            min_loss = 1e10
            patience_cnt = 0
            val_loss_values = []
            best_epoch = 0

            t = time.time()
            model.train()

            for epoch in range(self.epochs):
                loss_train = 0.0
                correct = 0
                for i, data in enumerate(train_loader):
                    optimizer.zero_grad()
                    data = data.to(self.device)
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
                data = data.to(self.device)
                out = model(data)
                pred = out.max(dim=1)[1]
                correct += pred.eq(data.y).sum().item()
                loss_test += F.nll_loss(out, data.y).item()

                # print(pred.numpy().tolist())  ###
                # print(data.y.numpy().tolist())

            return correct / len(loader.dataset), loss_test

        def compute_all(loader, pre: list, real: list):
            model.eval()
            correct = 0.0
            loss_test = 0.0

            for data in loader:
                data = data.to(self.device)
                out = model(data)
                pred = out.max(dim=1)[1]
                correct += pred.eq(data.y).sum().item()
                loss_test += F.nll_loss(out, data.y).item()

                pre.append(pred.item())
                real.append(data.y.item())

            return correct / len(loader.dataset), loss_test

        # Model training
        best_model = train()
        # Restore best model for test set
        model.load_state_dict(torch.load('{}.pth'.format(best_model)))
        # test_acc, test_loss = compute_test(test_loader)

        pre = []
        real = []
        test_acc, test_loss = compute_all(whole_loader, pre, real)
        print('Test set results, loss = {:.6f}, accuracy = {:.6f}'.format(test_loss, test_acc))
        print(pre)
        print(real)
        return pre, real, train_loss, valid_loss

    def pos_reg(self):
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.seed)

        # path = r'D:\pycharm\project\xtzz\data\output\plane'
        loader = MyDatasetLoader(self.measurenment)
        dataset = loader.get_dataset()
        self.num_classes = 16  # 一共多少种态势
        self.num_features = dataset[0].x.shape[1]

        num_training = int(len(dataset) * 0.8)
        num_val = int(len(dataset) * 0.1)
        num_test = len(dataset) - (num_training + num_val)
        training_set, validation_set, test_set = random_split(dataset, [num_training, num_val, num_test])

        train_loader = DataLoader(training_set, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(validation_set, batch_size=self.batch_size, shuffle=False)
        test_loader = DataLoader(test_set, batch_size=self.batch_size, shuffle=False)

        whole_loader = DataLoader(dataset)  ###

        model = Model(self).to(self.device)

        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        train_loss = []
        valid_loss = []

        def train():
            min_loss = 1e10
            patience_cnt = 0
            val_loss_values = []
            best_epoch = 0

            t = time.time()
            model.train()

            for epoch in range(self.epochs):
                loss_train = 0.0
                correct = 0
                for i, data in enumerate(train_loader):
                    optimizer.zero_grad()
                    data = data.to(self.device)
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
                data = data.to(self.device)
                out = model(data)
                pred = out.max(dim=1)[1]
                correct += pred.eq(data.y).sum().item()
                loss_test += F.nll_loss(out, data.y).item()

                # print(pred.numpy().tolist())  ###
                # print(data.y.numpy().tolist())

            return correct / len(loader.dataset), loss_test

        def compute_all(loader, pre: list, real: list):
            model.eval()
            correct = 0.0
            loss_test = 0.0

            for data in loader:
                data = data.to(self.device)
                if data.edge_index.equal(torch.tensor([], dtype=torch.int64)):  ###################
                    pred = torch.LongTensor([0])
                else:
                    out = model(data)
                    pred = out.max(dim=1)[1]

                correct += pred.eq(data.y).sum().item()
                loss_test += F.nll_loss(out, data.y).item()

                pre.append(pred.item())
                real.append(data.y.item())

            return correct / len(loader.dataset), loss_test

        # Model training
        best_model = train()
        # Restore best model for test set
        model.load_state_dict(torch.load('{}.pth'.format(best_model)))
        # test_acc, test_loss = compute_test(test_loader)

        pre = []
        real = []

        test_acc, test_loss = compute_all(whole_loader, pre, real)

        print('Test set results, loss = {:.6f}, accuracy = {:.6f}'.format(test_loss, test_acc))
        # print(pre)
        # print(real)

        return pre, real, train_loss, valid_loss

