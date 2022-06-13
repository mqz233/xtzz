from DataOperator.fluxdbOperator import fluxdbOperator
import os
import json
import torch
import numpy as np
from typing import List, Union
from torch_geometric.data import Data
from DataOperator.jsonOperator import jsonOperator as jo

Edge_Indices = List[Union[np.ndarray, None]]
Edge_Weights = List[Union[np.ndarray, None]]
Node_Features = List[Union[np.ndarray, None]]
Targets = List[Union[np.ndarray, None]]
Additional_Features = List[np.ndarray]


class DynamicGraphTemporalSignal(object):
    r"""A data iterator object to contain a dynamic graph with a
    changing edge set and weights . The feature set and node labels
    (target) are also dynamic. The iterator returns a single discrete temporal
    snapshot for a time period (e.g. day or week). This single snapshot is a
    Pytorch Geometric Data object. Between two temporal snapshots the edges,
    edge weights, target matrices and optionally passed attributes might change.

    Args:
        edge_indices (List of Numpy arrays): List of edge index tensors.
        edge_weights (List of Numpy arrays): List of edge weight tensors.
        features (List of Numpy arrays): List of node feature tensors.
        targets (List of Numpy arrays): List of node label (target) tensors.
        **kwargs (optional List of Numpy arrays): List of additional attributes.
    """

    def __init__(
        self,
        edge_indices: Edge_Indices,
        edge_weights: Edge_Weights,
        features: Node_Features,
        targets: Targets,
        **kwargs: Additional_Features
    ):
        self.edge_indices = edge_indices
        self.edge_weights = edge_weights
        self.features = features
        self.targets = targets
        self.additional_feature_keys = []
        for key, value in kwargs.items():
            setattr(self, key, value)
            self.additional_feature_keys.append(key)
        self._check_temporal_consistency()
        self._set_snapshot_count()

    def _check_temporal_consistency(self):
        assert len(self.features) == len(
            self.targets
        ), "Temporal dimension inconsistency."
        assert len(self.edge_indices) == len(
            self.edge_weights
        ), "Temporal dimension inconsistency."
        assert len(self.features) == len(
            self.edge_weights
        ), "Temporal dimension inconsistency."
        for key in self.additional_feature_keys:
            assert len(self.targets) == len(
                getattr(self, key)
            ), "Temporal dimension inconsistency."

    def _set_snapshot_count(self):
        self.snapshot_count = len(self.features)

    def _get_edge_index(self, time_index: int):
        if self.edge_indices[time_index] is None:
            return self.edge_indices[time_index]
        else:
            return torch.LongTensor(self.edge_indices[time_index])

    def _get_edge_weight(self, time_index: int):
        if self.edge_weights[time_index] is None:
            return self.edge_weights[time_index]
        else:
            return torch.FloatTensor(self.edge_weights[time_index])

    def _get_features(self, time_index: int):
        if self.features[time_index] is None:
            return self.features[time_index]
        else:
            return torch.FloatTensor(self.features[time_index])

    def _get_target(self, time_index: int):
            return torch.LongTensor(self.targets[time_index])


    def _get_additional_feature(self, time_index: int, feature_key: str):
        feature = getattr(self, feature_key)[time_index]
        if feature.dtype.kind == "i":
            return torch.LongTensor(feature)
        elif feature.dtype.kind == "f":
            return torch.FloatTensor(feature)

    def _get_additional_features(self, time_index: int):
        additional_features = {
            key: self._get_additional_feature(time_index, key)
            for key in self.additional_feature_keys
        }
        return additional_features

    def __getitem__(self, time_index):
        x = self._get_features(time_index)
        edge_index = self._get_edge_index(time_index)
        edge_weight = self._get_edge_weight(time_index)
        y = self._get_target(time_index)
        additional_features = self._get_additional_features(time_index)

        snapshot = Data(x=x, edge_index=edge_index, edge_attr=edge_weight,
                        y=y, **additional_features)
        return snapshot

    def __next__(self):
        if self.t < len(self.features):
            snapshot = self[self.t]
            self.t = self.t + 1
            return snapshot
        else:
            self.t = 0
            raise StopIteration

    def __iter__(self):
        self.t = 0
        return self

    def __len__(self):
        return len(self.targets)

class MyDatasetLoader:


    def __init__(self,measurement):

        self._read_data(measurement)

    def _read_data(self,measurement):

        self._dataset = []
        client = fluxdbOperator()
        result = client.select_num_battle(measurement)
        # for i in range(len(result)):
        #     del result[i]['time']
        #     del result[i]['sences']
        #     del result[i]['frameId']
        #     del result[i]['Time']
        #     # del result[i]['stage']
        #     # del result[i]['eval']
        #     for key in result[i].keys():
        #         result[i][key] = json.loads(result[i][key])
        self._dataset = result

    def _get_edges(self,name:str):

        edges = []
        for i in range(len(self._dataset)):
            matrix1 = np.array(json.loads(self._dataset[i]['radarList']))
            matrix2 = np.array(json.loads(self._dataset[i]['locked']))
            matrix3 = np.array(json.loads(self._dataset[i]['atkList']))
            matrix4 = np.array(json.loads(self._dataset[i]['conList']))
            matrix5 = np.array(json.loads(self._dataset[i]['comm']))
            matrix6 = np.array(json.loads(self._dataset[i]['suppressList']))
            matrix7 = np.array(json.loads(self._dataset[i]['echo']))
            matrix = matrix1+matrix2+matrix3+matrix4+matrix5+matrix6+matrix7
            edges.append([])
            for x in range(matrix.shape[0]):
                for y in range(matrix.shape[1]):
                    if x != y and matrix[x][y] != 0:
                        edges[i].append([x,y])
            # if not edges[i]:
            #     edges[i].append([])
            #     edges[i].append([])
            edges[i] = np.array(edges[i]).T

        return edges

    def _get_edge_weights(self,name:str):

        edge_weights = []
        for i in range(len(self._dataset)):
            matrix1 = np.array(json.loads(self._dataset[i]['radarList']))
            matrix2 = np.array(json.loads(self._dataset[i]['locked']))
            matrix3 = np.array(json.loads(self._dataset[i]['atkList']))
            matrix4 = np.array(json.loads(self._dataset[i]['conList']))
            matrix5 = np.array(json.loads(self._dataset[i]['comm']))
            matrix6 = np.array(json.loads(self._dataset[i]['suppressList']))
            matrix7 = np.array(json.loads(self._dataset[i]['echo']))
            matrix = matrix1 + matrix2 + matrix3 + matrix4 + matrix5 + matrix6 + matrix7
            k = 0
            for x in range(matrix.shape[0]):
                for y in range(matrix.shape[1]):
                    if x != y and matrix[x][y] != 0:
                        k=k+1
            edge_weights.append(np.array([1 for d in range(k)]).T)
        return edge_weights

    def _get_features(self):
        diclist = self._dataset
        planes = [int(n) for n in range(len(json.loads(diclist[0]['name'])))]  #多少架飞机
        dataset_length = len(diclist)
        # feature_num = 49+len(planes)*9
        feature_num = 21
        features = np.zeros((dataset_length, len(planes), feature_num))
        list = [
                'svv',
                # 'isRed',
                # 'type',
                # 'value',
                # 'ra_Pro_Angle',
                # 'ra_Pro_Radius',
                # 'ra_StartUp_Delay',
                # 'ra_Detect_Delay',
                # 'ra_Process_Delay',
                # 'ra_FindTar_Delay',
                # 'ra_Rang_Accuracy',
                # 'ra_Angle_Accuracy',
                # 'MisMaxAngle',
                # 'MisMaxRange',
                # 'MisMinDisescapeDis',
                # 'MisMaxDisescapeDis',
                # 'MisMaxV',
                # 'MisMaxOver',
                # 'MisLockTime',
                # 'MisHitPro',
                # 'MisMinAtkDis',
                # 'MisNum',
                # 'EchoInitState',
                # 'EchoFackTarNum',
                # 'EchoDis',
                # 'SupInitState',
                # 'SupTarNum',
                # 'SupMinDis',
                # 'SupMaxAngle',
                'posx',
                'posy',
                'posz',
                'v',
                'Vn',
                'Vu',
                'Ve',
                'yaw',
                'pitch',
                'roll',
                'radar_flag',
                'rsuppress_flag',
                'echo_flag',
                'targetNum',
                'radar_radius',
                'atcNum',
                'controlNum',
                'comNum',
                'suppressNum',
                'echoNum',
                # 'radarList',
                # 'locked',
                # 'det_pro',
                # 'range_acc',
                # 'angle_acc',
                # 'atkList',
                # 'conList',
                # 'suppressList',
                # 'echo'
                  ]
        for frame in range(len(diclist)):
            for plane in range(len(planes)):
                feature = []
                # for i in range(len(list) - 9):
                for i in range(len(list)):
                    feature.append(json.loads(diclist[frame][list[i]])[plane])
                # for i in range(len(planes)):
                #     feature.append(json.loads(diclist[frame]['radarList'])[plane][i])
                # for i in range(len(planes)):
                #     feature.append(json.loads(diclist[frame]['locked'])[plane][i])
                # for i in range(len(planes)):
                #     feature.append(json.loads(diclist[frame]['det_pro'])[plane][i])
                # for i in range(len(planes)):
                #     feature.append(json.loads(diclist[frame]['range_acc'])[plane][i])
                # for i in range(len(planes)):
                #     feature.append(json.loads(diclist[frame]['angle_acc'])[plane][i])
                # for i in range(len(planes)):
                #     feature.append(json.loads(diclist[frame]['atkList'])[plane][i])
                # for i in range(len(planes)):
                #     feature.append(json.loads(diclist[frame]['conList'])[plane][i])
                # for i in range(len(planes)):
                #     feature.append(json.loads(diclist[frame]['suppressList'])[plane][i])
                # for i in range(len(planes)):
                #     feature.append(json.loads(diclist[frame]['echo'])[plane][i])
                features[frame, plane, :] = feature

        self.features = [
            features[i, :]
            for i in range(len(features))
        ]

    def _get_targets(self):
        # target eoncoding: {0 : 'Grasp', 1 : 'Move', 2 : 'Negative',
        #                   3 : 'Position', 4 : 'Reach', 5 : 'Release'}
        # targets = []
        # for _, y in self._dataset["LABEL"].items():
        #     a = []
        #     a.append(y)
        #     targets.append(a)
        #
        # self.targets = [
        #     targets[i]     ### 识 别
        #     for i in range(len(targets))
        # ]
        dic = {
            'A1': 0,
            'A2': 1,
            'A3': 2,
            'A4': 3,
            'B1': 4,
            'B2': 5,
            'B3': 6,
            'B4': 7,
            'C1': 8,
            'C2': 9,
            'C3': 10,
            'C4': 11,
            'D1': 12,
            'D2': 13,
            'D3': 14,
            'D4': 15
        }
        self.targets = []
        for i in range(len(self._dataset)):
            for key in dic.keys():
                pos = self._dataset[i]['stage'] + self._dataset[i]['eval']
                if pos == key:
                    self.targets.append([dic[key]])


    def get_dataset(self) :

        self._get_features()
        self._get_targets()
        comm = DynamicGraphTemporalSignal(
            self._get_edges("comm"),self._get_edge_weights("comm"), self.features, self.targets
        )

        return comm


