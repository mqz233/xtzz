from typing import List
import json

from six.moves import urllib
# from torch_geometric_temporal.signal import StaticGraphTemporalSignal
import torch
import numpy as np
from typing import List, Union
from torch_geometric.data import Data


Edge_Index = Union[np.ndarray, None]
Edge_Weight = Union[np.ndarray, None]
Node_Features = List[Union[np.ndarray, None]]
Targets = List[Union[np.ndarray, None]]
Additional_Features = List[np.ndarray]


class StaticGraphTemporalSignal(object):
    r"""A data iterator object to contain a static graph with a dynamically
    changing constant time difference temporal feature set (multiple signals).
    The node labels (target) are also temporal. The iterator returns a single
    constant time difference temporal snapshot for a time period (e.g. day or week).
    This single temporal snapshot is a Pytorch Geometric Data object. Between two
    temporal snapshots the features and optionally passed attributes might change.
    However, the underlying graph is the same.

    Args:
        edge_index (Numpy array): Index tensor of edges.
        edge_weight (Numpy array): Edge weight tensor.
        features (List of Numpy arrays): List of node feature tensors.
        targets (List of Numpy arrays): List of node label (target) tensors.
        **kwargs (optional List of Numpy arrays): List of additional attributes.
    """

    def __init__(
        self,
        edge_index: Edge_Index,
        edge_weight: Edge_Weight,
        features: Node_Features,
        targets: Targets,
        **kwargs: Additional_Features
    ):
        self.edge_index = edge_index
        self.edge_weight = edge_weight
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
        for key in self.additional_feature_keys:
            assert len(self.targets) == len(
                getattr(self, key)
            ), "Temporal dimension inconsistency."

    def _set_snapshot_count(self):
        self.snapshot_count = len(self.features)

    def _get_edge_index(self):
        if self.edge_index is None:
            return self.edge_index
        else:
            return torch.LongTensor(self.edge_index)

    def _get_edge_weight(self):
        if self.edge_weight is None:
            return self.edge_weight
        else:
            return torch.FloatTensor(self.edge_weight)

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

    def __getitem__(self, time_index: int):
        x = self._get_features(time_index)
        edge_index = self._get_edge_index()
        edge_weight = self._get_edge_weight()
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

class MTMDatasetLoader:
    """
    A dataset of `Methods-Time Measurement-1 <https://en.wikipedia.org/wiki/Methods-time_measurement>`_
    (MTM-1) motions, signalled as consecutive video frames of 21 3D hand keypoints, acquired via
    `MediaPipe Hands <https://google.github.io/mediapipe/solutions/hands.html>`_ from RGB-Video
    material. Vertices are the finger joints of the human hand and edges are the bones connecting
    them. The targets are manually labeled for each frame, according to one of the five MTM-1
    motions (classes :math:`C`): Grasp, Release, Move, Reach, Position plus a negative class for
    frames without graph signals (no hand present). This is a classification task where :math:`T`
    consecutive frames need to be assigned to the corresponding class :math:`C`. The data x is
    returned in shape :obj:`(3, 21, T)`, the target is returned one-hot-encoded in shape :obj:`(T, 6)`.
    """

    def __init__(self):
        self._read_web_data()

    def _read_web_data(self):
        # url = "https://raw.githubusercontent.com/benedekrozemberczki/pytorch_geometric_temporal/master/dataset/mtm_1.json"
        # self._dataset = json.loads(urllib.request.urlopen(url).read())

        with open('HGP/mtm_1.json', 'r') as fr:
            self._dataset = json.loads(fr.read())

    def _get_edges(self):
        self._edges = np.array(self._dataset["edges"]).T

    def _get_edge_weights(self):
        self._edge_weights = np.array([1 for d in self._dataset["edges"]]).T

    def _get_features(self):
        dic = self._dataset
        joints = [str(n) for n in range(21)]
        dataset_length = len(dic["0"].values())
        features = np.zeros((dataset_length, 21, 3))

        for j, joint in enumerate(joints):
            for t, xyz in enumerate(dic[joint].values()):
                xyz_tuple = list(map(float, xyz.strip("()").split(",")))
                features[t, j, :] = xyz_tuple

        self.features = [
            features[i, :]
            for i in range(len(features))
        ]

    def _get_targets(self):
        # target eoncoding: {0 : 'Grasp', 1 : 'Move', 2 : 'Negative',
        #                   3 : 'Position', 4 : 'Reach', 5 : 'Release'}
        targets = []
        for _, y in self._dataset["LABEL"].items():
            a = []
            a.append(y)
            targets.append(a)

        self.targets = [
            targets[i]     ### 识 别
            for i in range(len(targets))
        ]

    def get_dataset(self, frames: int = 16) -> StaticGraphTemporalSignal:
        """Returning the MTM-1 motion data iterator.

        Args types:
            * **frames** *(int)* - The number of consecutive frames T, default 16.
        Return types:
            * **dataset** *(StaticGraphTemporalSignal)* - The MTM-1 dataset.
        """
        self.frames = frames
        self._get_edges()
        self._get_edge_weights()
        self._get_features()
        self._get_targets()

        dataset = StaticGraphTemporalSignal(
            self._edges, self._edge_weights, self.features, self.targets
        )
        return dataset
