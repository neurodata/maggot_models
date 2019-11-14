#%%
import math
import os
from operator import itemgetter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from graspy.cluster import GaussianCluster
from graspy.embed import AdjacencySpectralEmbed, OmnibusEmbed
from graspy.models import SBMEstimator
from graspy.plot import heatmap, pairplot
from graspy.utils import binarize, cartprod, pass_to_ranks
from joblib.parallel import Parallel, delayed
from matplotlib.colors import LogNorm
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from spherecluster import SphericalKMeans

from src.data import load_everything, load_networkx
from src.utils import savefig, meta_to_array
from src.visualization import sankey

from node2vec import Node2Vec

#%%
MB_VERSION = "mb_2019-09-23"
BRAIN_VERSION = "2019-09-18-v2"
GRAPH_TYPES = ["Gad", "Gaa", "Gdd", "Gda"]
GRAPH_TYPE_LABELS = [r"A $\to$ D", r"A $\to$ A", r"D $\to$ D", r"D $\to$ A"]
N_GRAPH_TYPES = len(GRAPH_TYPES)

FNAME = os.path.basename(__file__)[:-3]
print(FNAME)

adj, class_labels, side_labels = load_everything(
    "G", version=BRAIN_VERSION, return_class=True, return_side=True
)

graph = load_networkx("G", BRAIN_VERSION)
node2vec = Node2Vec(
    graph, dimensions=6, workers=12, p=0.5, q=0.5, walk_length=100, num_walks=20
)


model = node2vec.fit(window=20, min_count=1, batch_words=4)
vecs = [model.wv.get_vector(n) for n in graph.node()]

embedding = np.array(vecs)

pairplot(embedding, labels=meta_to_array(graph, "Class"), palette="tab20")

