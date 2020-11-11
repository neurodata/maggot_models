#%%
import os
import pickle
import time
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from anytree import LevelOrderGroupIter
from scipy.sparse import dia_matrix, diags, load_npz, save_npz
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import adjusted_rand_score

from graspologic.cluster import AutoGMMCluster
from graspologic.embed import AdjacencySpectralEmbed
from graspologic.plot import pairplot
from src.io import savefig
from src.visualization import set_theme

np.random.seed(88888)
set_theme()


FNAME = os.path.basename(__file__)[:-3]

RECOMPUTE = False


def stashfig(name, **kws):
    savefig(name, foldername=FNAME, save_on=True, **kws)


def read_if_available(file_name, func_if_unavailable, recompute=RECOMPUTE, npz=False):
    if os.path.isfile(file_name) and not recompute:
        if npz:
            out = load_npz(file_name)
        else:
            with open(file_name, "rb") as f:
                out = pickle.load(f)
        print(f"Read file from {file_name}")
    else:
        out = func_if_unavailable()
        if npz:
            save_npz(file_name, out, compressed=True)
        else:
            with open(file_name, "wb") as f:
                pickle.dump(out, f)
        print(f"Wrote file to {file_name}")
    return out


#%%
data_dir = Path("maggot_models/data/processed/2020-09-23")
connector_loc = data_dir / "connectors.csv"
connectors = pd.read_csv(connector_loc, index_col=0)

out_dir = Path(f"maggot_models/notebooks/outs/{FNAME}/outs/")


#%%
# Throw out a bunch of stuff to make problem more tractable
connectors = connectors[connectors["in_subgraph"]]
connectors = connectors[
    (connectors["presynaptic_type"] == "axon")
    & (connectors["postsynaptic_type"] == "dendrite")
]
n_subsample = None  # 2 ** 16
if n_subsample:
    choice_inds = np.random.choice(len(connectors), size=n_subsample, replace=False)
    connectors = connectors.iloc[choice_inds]
print(f"Using {len(connectors)} connectors")

#%%
graph = nx.from_pandas_edgelist(
    connectors,
    source="presynaptic_to",
    target="postsynaptic_to",
    edge_attr=True,
    create_using=nx.MultiDiGraph(),
    edge_key="connector_id",
)


#%%
line_graph_file = out_dir / "line-graph.pkl"


def make_line_graph():
    currtime = time.time()
    line_graph = nx.line_graph(graph, create_using=nx.DiGraph)
    print(f"{time.time() - currtime} elapsed to make line graph")
    return line_graph


line_graph = read_if_available(line_graph_file, make_line_graph)

#%%
outdeg = line_graph.out_degree()
to_remove = [n for n, deg in outdeg if deg <= 10]
line_graph.remove_nodes_from(to_remove)

indeg = line_graph.in_degree()
to_remove = [n for n, deg in indeg if deg <= 10]
line_graph.remove_nodes_from(to_remove)


#%%
sparse_adj_file = out_dir / "sparse-line-adj.npz"

nodelist = list(sorted(line_graph.nodes))


def make_sparse_adj():
    currtime = time.time()
    line_graph_adj = nx.to_scipy_sparse_matrix(line_graph, nodelist=nodelist)
    print(f"{time.time() - currtime} elapsed to make sparse adjacency")
    return line_graph_adj


line_graph_adj = read_if_available(
    sparse_adj_file, make_sparse_adj, npz=True, recompute=True
)

#%%
out_degree = np.squeeze(np.array(line_graph_adj.sum(axis=1), dtype=np.float64))
in_degree = np.squeeze(np.array(line_graph_adj.sum(axis=0), dtype=np.float64))
mean_total_degree = np.mean(out_degree)
regularizer = 0.5 * mean_total_degree
out_degree += regularizer
in_degree += regularizer

inv_out_degree = 1 / np.sqrt(out_degree)
inv_in_degree = 1 / np.sqrt(in_degree)

inv_out_degree = diags(inv_out_degree)
inv_in_degree = diags(inv_in_degree)

line_graph_lap = inv_out_degree @ line_graph_adj @ inv_in_degree

#%%
tsvd = TruncatedSVD(n_components=8, algorithm="randomized")
U = tsvd.fit_transform(line_graph_lap)
S = tsvd.singular_values_
Vt = tsvd.components_

U = inv_out_degree @ U
#%%
n_subsample = 2 ** 14
choice_inds = np.random.choice(U.shape[0], size=n_subsample, replace=False)
U_subsample = U[choice_inds, :]


pairplot(U_subsample, marker="o", size=1)
stashfig("U-pairplot")


# %%
