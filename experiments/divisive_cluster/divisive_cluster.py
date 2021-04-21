#%%
import datetime
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.cluster.hierarchy import fcluster
from scipy.cluster.hierarchy import linkage as linkage_cluster
from scipy.spatial.distance import squareform
from sklearn.metrics.pairwise import pairwise_distances

from giskard.plot import crosstabplot, dissimilarity_clustermap
from graspologic.plot.plot_matrix import scattermap
from graspologic.utils import symmetrize
from src.data import load_maggot_graph, join_node_meta
from src.io import savefig
from src.visualization import CLASS_COLOR_DICT as palette
from src.visualization import adjplot, set_theme

t0 = time.time()


def stashfig(name, **kws):
    savefig(
        name,
        pathname="./maggot_models/experiments/agglomerative_cluster/figs",
        **kws,
    )


out_dir = Path(__file__).parent / "outs"
embedding_loc = Path("maggot_models/experiments/embed/outs/stage2_embedding.csv")

#%% load the embedding, get the correct subset of data
embedding_df = pd.read_csv(embedding_loc, index_col=0)
embedding_df = embedding_df.groupby(embedding_df.index).mean()
mg = load_maggot_graph()
nodes = mg.nodes.copy()
nodes = nodes[nodes.index.isin(embedding_df.index)]
nodes = nodes[nodes["paper_clustered_neurons"]]
embedding_df = embedding_df[embedding_df.index.isin(nodes.index)]
nodes = nodes.reindex(embedding_df.index)
embedding = embedding_df.values

# %% [markdown]
# ## Clustering

from graspologic.cluster import DivisiveCluster

# parameters
n_levels = 10  # max # of splits in the recursive clustering
metric = "bic"  # metric on which to decide best split
n_components = 8

X = embedding[:, :n_components]

dc = DivisiveCluster()

#%%
# params = [
#     {"d": 8, "bic_ratio": 0, "min_split": 32},
#     {"d": 8, "bic_ratio": 0.95, "min_split": 32},
# ]

# for p in params:
#     print(p)
#     d = p["d"]
#     bic_ratio = p["bic_ratio"]
#     min_split = p["min_split"]
#     X = embedding[:, :d]
#     basename = f"-d={d}-bic_ratio={bic_ratio}-min_split={min_split}"

#     currtime = time.time()
#     np.random.seed(8888)
#     mc = BinaryCluster(
#         "0",
#         n_init=50,  # number of initializations for GMM at each stage
#         meta=nodes,  # stored for plotting and adding labels
#         X=X,  # input data that actually matters
#         bic_ratio=bic_ratio,
#         reembed=False,
#         min_split=min_split,
#     )

#     mc.fit(n_levels=n_levels, metric=metric)
#     print(f"{(time.time() - currtime)/60:0.2f} minutes elapsed for clustering")

#     cluster_meta = mc.meta

#     # save results
#     cluster_meta.to_csv("meta" + basename)

#     print()

#%%
elapsed = time.time() - t0
delta = datetime.timedelta(seconds=elapsed)
print("----")
print(f"Script took {delta}")
print(f"Completed at {datetime.datetime.now()}")
print("----")