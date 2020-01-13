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

from src.data import load_everything, load_networkx, load_metagraph
from src.utils import savefig, meta_to_array
from src.visualization import sankey

from node2vec import Node2Vec
import colorcet as cc

#%%
BRAIN_VERSION = "2019-09-18-v2"

FNAME = os.path.basename(__file__)[:-3]
print(FNAME)

# %% [markdown]
# #

mg = load_metagraph("Gadn")
mg.make_lcc()
graph = mg.g
node2vec = Node2Vec(
    graph, dimensions=8, workers=12, p=2, q=0.5, walk_length=32, num_walks=1024
)
model = node2vec.fit(window=20, min_count=1, batch_words=4)
nodedata_list = list(graph.nodes(data=True))
vecs = [model.wv.get_vector(str(n)) for n, _ in nodedata_list]
embedding = np.array(vecs)
class_labels = [d["Merge Class"] for _, d in nodedata_list]
# %% [markdown]
# #
n_unique = len(np.unique(class_labels))

pairplot(embedding, labels=class_labels, palette=cc.glasbey_light[:n_unique])

# %% [markdown]
# #
plot_df = pd.DataFrame(data=embedding)
plot_df["Label"] = class_labels
n_components = 6
for i in range(n_components):
    for j in range(i + 1, n_components):
        fig, ax = plt.subplots(1, 1, figsize=(20, 20))
        sns.scatterplot(
            data=plot_df,
            x=i,
            y=j,
            hue="Label",
            palette=cc.glasbey_light[:n_unique],
            s=20,
            ax=ax,
        )

