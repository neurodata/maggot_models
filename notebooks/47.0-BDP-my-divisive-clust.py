# %% [markdown]
# #
import json
import os
import pickle
import warnings
from copy import deepcopy
from operator import itemgetter
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import pydot
import seaborn as sns

# %% [markdown]
# #
# %% [markdown]
# #
from anytree import (
    LevelOrderGroupIter,
    LevelOrderIter,
    NodeMixin,
    PostOrderIter,
    RenderTree,
)

# %% [markdown]
# #
from anytree.exporter import DotExporter
from anytree.util import leftsibling
from joblib import Parallel, delayed
from joblib.parallel import Parallel, delayed
from scipy.cluster.hierarchy import dendrogram
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score, silhouette_score
from spherecluster import SphericalKMeans

from graspy.cluster import AutoGMMCluster, GaussianCluster
from graspy.embed import AdjacencySpectralEmbed, OmnibusEmbed
from graspy.models import DCSBMEstimator, SBMEstimator
from graspy.plot import heatmap, pairplot
from graspy.utils import binarize, cartprod, get_lcc, pass_to_ranks
from src.data import load_everything
from src.embed import lse
from src.hierarchy import signal_flow
from src.io import stashfig
from src.utils import export_skeleton_json, savefig
from src.visualization import clustergram, palplot, sankey

# %% [markdown]
# #


centers = np.array(
    [
        [0, 0],
        [10, 0],
        [0, 10],
        [30, 30],
        [20, 30],
        [30, 20],
        [200, 200],
        [210, 200],
        [200, 210],
        [230, 230],
        [220, 230],
        [230, 220],
    ]
)
X, y = make_blobs(n_samples=500, n_features=2, centers=centers, cluster_std=1)
X_df = pd.DataFrame(
    data=np.concatenate((X, y[:, np.newaxis]), axis=-1),
    columns=("Dim 0", "Dim 1", "Labels"),
)


sns.scatterplot(
    data=X_df, x="Dim 0", y="Dim 1", hue="Labels", palette="Set1", legend=False
)
# %% [markdown]
# #


class PartitionCluster(NodeMixin):
    def __init__(self):
        self.min_split_samples = 5

    def fit(self, X, y=None):
        n_samples = X.shape[0]

        if n_samples > self.min_split_samples:
            cluster = GaussianCluster(min_components=1, max_components=2, n_init=20)
            cluster.fit(X)
            self.model_ = cluster
        else:
            self.pred_labels_ = np.zeros(X.shape[0])
            self.left_ = None
            self.right_ = None
            self.model_ = None
            return self

        # recurse
        if cluster.n_components_ != 1:
            pred_labels = cluster.predict(X)
            self.pred_labels_ = pred_labels
            indicator = pred_labels == 0
            self.X_left_ = X[indicator, :]
            self.X_right_ = X[~indicator, :]
            split_left = PartitionCluster()
            self.left_ = split_left.fit(self.X_left_)

            split_right = PartitionCluster()
            self.right_ = split_right.fit(self.X_right_)
        else:
            self.pred_labels_ = np.zeros(X.shape[0])
            self.left_ = None
            self.right_ = None
            self.model_ = None
        return self

    # def predict_sample(sample_ind):
    #     direction = self.pred_labels_[sample_ind]
    #     if direction == 0:
    #         self.predict_sample(sample_ind)

    # def predict(self, X, y=None):
    #     if self.left_ is None:
    #         print("Stop")
    #         return X.shape[0] * [[-1]]

    #     else:
    #         # n_samples = X.shape[0]
    #         indicator = self.pred_labels_ == 0
    #         left_pred = self.left_.predict(self.X_left_)
    #         right_pred = self.right_.predict(self.X_right_)
    #         full_pred_labels = []  # np.empty_like(self.pred_labels_)
    #         left_counter = 0
    #         right_counter = 0
    #         print(indicator)
    #         for i, went_left in enumerate(indicator):
    #             if went_left:
    #                 old_list = left_pred[left_counter]
    #                 old_list.insert(0, 0)
    #                 full_pred_labels.append(old_list)
    #                 left_counter += 1
    #             else:
    #                 old_list = right_pred[right_counter]
    #                 old_list.insert(0, 1)
    #                 full_pred_labels.append(old_list)
    #                 right_counter += 1
    #         return full_pred_labels
    # def predict(self, X, y=None):
    #     pred_labels = self.pred_labels_
    #     sample_preds = []
    #     for i, sample in enumerate(X):
    #         if self.left_ is None:
    #             return X.shape[0] * [[-1]]
    #         sample = np.array([sample])
    #         if pred_labels[i] == 0:
    #             sample_pred = 0
    #             p = self.left_.predict(sample)[0]
    #             if p == [-1]:
    #                 p = [0]
    #                 print("Leaf")
    #         else:
    #             sample_pred = 1
    #             p = self.right_.predict(sample)[0]
    #             if p == [-1]:
    #                 p = [1]
    #         total_pred = [sample_pred] + p
    #         sample_preds.append(total_pred)
    #     return sample_preds
    # def predict(self, X, y=None):
    #     pred_labels = pclust.pred_labels_
    #     indicator = pred_labels == 0
    #     left = pclust.left_
    #     right = pclust.right_

    #     pred_strs = []
    #     for i, val in enumerate(pred_labels):
    #         pred_strs.append(f"{val}")

    #     print(pred_strs)
    #     # if left is not None:

    def predict_sample(self, sample):
        model = self.model_
        if model is not None:
            pred = model.predict([sample])[0]
            if pred == 0:
                next_pred = self.left_.predict_sample(sample)
            if pred == 1:
                next_pred = self.right_.predict_sample(sample)
            total_pred = str(pred) + next_pred
        else:
            total_pred = ""
        return total_pred

    def predict(self, X):
        predictions = []
        for sample in X:
            pred = self.predict_sample(sample)
            predictions.append(pred)
        return np.array(predictions)


pclust = PartitionCluster()
pclust.fit(X)
preds = pclust.predict(X)
# %% [markdown]
# #


class DivisiveCluster(NodeMixin):
    def __init__(self, name="", min_split_samples=5, parent=None, children=None):
        self.name = name
        self.parent = parent
        if children:
            self.children = children
        self.min_split_samples = min_split_samples
        self.samples_ = None
        self.y_ = None

    def fit(self, X, y=None):
        n_samples = X.shape[0]
        self.n_samples_ = n_samples
        if n_samples > self.min_split_samples:
            cluster = GaussianCluster(min_components=1, max_components=2, n_init=40)
            cluster.fit(X)
            pred_labels = cluster.predict(X)
            self.pred_labels_ = pred_labels
            self.model_ = cluster
            if cluster.n_components_ != 1:
                indicator = pred_labels == 0
                self.X_children_ = (X[indicator, :], X[~indicator, :])
                children = []
                for i, X_child in enumerate(self.X_children_):
                    child = DivisiveCluster(name=self.name + str(i), parent=self)
                    child = child.fit(X_child)
                    children.append(child)
                self.children = children
        return self

    def predict_sample(self, sample, label):
        if not self.children:
            if not self.samples_:
                self.samples_ = []
            self.samples_.append(sample)
            if not self.y_:
                self.y_ = []
            self.y_.append(label)
            return self
        else:
            pred = self.model_.predict([sample])[0]
            if pred == 0:
                return self.children[0].predict_sample(sample, label)
            else:
                return self.children[1].predict_sample(sample, label)

    def print_tree(self):
        for pre, _, node in RenderTree(dc):
            treestr = "%s%s (%s)" % (pre, node.name, node.n_samples_)
            print(treestr.ljust(8))

    def build_linkage(self):
        # get a tuple of node at each level
        levels = []
        for group in LevelOrderGroupIter(dc):
            levels.append(group)

        # just find how many nodes are leaves
        # this is necessary only because we need to add n to non-leaf clusters
        num_leaves = 0
        for node in PostOrderIter(dc):
            if not node.children:
                num_leaves += 1

        link_count = 0
        node_index = 0
        linkages = []
        labels = []

        for g, group in enumerate(levels[::-1][:-1]):
            for i in range(len(group) // 2):
                # get partner nodes
                left_node = group[2 * i]
                right_node = group[2 * i + 1]
                # just double check that these are always partners
                assert leftsibling(right_node) == left_node

                # check if leaves, need to add some new fields to track for linkage
                if not left_node.children:
                    left_node._ind = node_index
                    left_node._n_clusters = 1
                    node_index += 1
                    labels.append(left_node.name)

                if not right_node.children:
                    right_node._ind = node_index
                    right_node._n_clusters = 1
                    node_index += 1
                    labels.append(right_node.name)

                # find the parent, count samples
                parent_node = left_node.parent
                n_clusters = left_node._n_clusters + right_node._n_clusters
                parent_node._n_clusters = n_clusters

                # assign an ind to this cluster for the dendrogram
                parent_node._ind = link_count + num_leaves
                link_count += 1

                # add a row to the linkage matrix
                linkages.append([left_node._ind, right_node._ind, g + 1, n_clusters])

        return (
            np.array(linkages, dtype=np.double),
            labels,
        )  # needs to be a double for scipy

        # split_left = PartitionCluster()
        # self.left_ = split_left.fit(self.X_left_)

        # split_right = PartitionCluster()
        # self.right_ = split_right.fit(self.X_right_)
        # else:
        #     self.pred_labels_ = np.zeros(X.shape[0])
        #     self.left_ = None
        #     self.right_ = None
        #     self.model_ = None
        #     return self

        # # recurse

        # else:
        #     self.pred_labels_ = np.zeros(X.shape[0])
        #     self.left_ = None
        #     self.right_ = None
        #     self.model_ = None

    # def predict(self, X, y=None):
    #     if not self.children_:
    #         return np.zeros(X.shape[0])

    #     pred_labels = self.model_.predict(X)
    #     pred_labels = pred_labels.astype(str)

    #     indicators = []
    #     indicators.append(pred_labels == 0)
    #     indicators.append(pred_labels == 1)
    #     for child, indicator in zip(self.children_, indicators):
    #         child_pred = child.predict(X[indicator], y[indicator])


#%%
dc = DivisiveCluster()
dc = dc.fit(X)


for node in PostOrderIter(dc):
    if not node.children:
        print("At leaf")
        print(f"{node.name}")
        print(f"{node.n_samples_}")
        print()


fig, ax = plt.subplots(1, 1, figsize=(5, 10))
dendrogram(
    linkage_mat,
    orientation="left",
    labels=labels,
    color_threshold=0,
    above_threshold_color="k",
    ax=ax,
)
ax.xaxis.set_visible(False)
ax.set_frame_on(False)
# %% [markdown]
# #
nodes = []
for sample, label in zip(X, y):
    nodes.append(dc.predict_sample(sample, label))


DotExporter(dc).to_dotfile("test_dot.dot")
dot_graph = pydot.graph_from_dot_file("test_dot.dot")[0]
g = nx.drawing.nx_pydot.from_pydot(dot_graph)
# A = nx.nx_agraph.to_agraph(g)
pos = nx.drawing.nx_agraph.pygraphviz_layout(g, prog="dot")
nx.draw(g, pos, with_labels=True)
# g = nx.drawing.nx_agraph.read_dot("test_dot.dot")

# get some ordering of the leaf nodes based on one of the anytree iterators
# plot the barplot with this custom ordering
#
