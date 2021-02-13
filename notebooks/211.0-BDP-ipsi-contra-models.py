#%%
from src.data import load_metagraph

import os

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns

import pymaid
from graspologic.utils import binarize, remove_loops
from src.data import load_metagraph
from src.graph import MetaGraph
from src.io import savefig
from src.pymaid import start_instance
from src.visualization import set_theme

start_instance()


FNAME = os.path.basename(__file__)[:-3]
print(FNAME)


def stashfig(name, **kws):
    savefig(name, foldername=FNAME, save_on=True, print_out=False, **kws)


set_theme()

#%%

from src.utils import get_paired_inds

mg = load_metagraph("G")
mg = mg.make_lcc()
meta = mg.meta.copy()
meta["inds"] = range(len(meta))
adj = mg.adj.copy()

# inds on left and right that correspond to the bilateral pairs
# here these are the union of true and predicted paris, though
lp_inds, rp_inds = get_paired_inds(meta, check_in=True)
lrp_inds = np.concatenate((lp_inds, rp_inds))
sided_meta = meta.iloc[lrp_inds].copy()
sided_adj = adj[np.ix_(lrp_inds, lrp_inds)]

LL = adj[np.ix_(lp_inds, lp_inds)]
LR = adj[np.ix_(lp_inds, rp_inds)]
RR = adj[np.ix_(rp_inds, rp_inds)]
RL = adj[np.ix_(rp_inds, lp_inds)]

from src.visualization import adjplot

adjplot(sided_adj, meta=sided_meta, plot_type="scattermap", sizes=(1, 1))

#%%
from graspologic.embed import AdjacencySpectralEmbed
from graspologic.utils import binarize, pass_to_ranks


def preprocess(A):
    return binarize(A)


n_components = 6
ase = AdjacencySpectralEmbed(n_components=n_components)
LL_embed_out, LL_embed_in = ase.fit_transform(preprocess(LL))
LR_embed_out, LR_embed_in = ase.fit_transform(preprocess(LR))
RR_embed_out, RR_embed_in = ase.fit_transform(preprocess(RR))
RL_embed_out, RL_embed_in = ase.fit_transform(preprocess(RL))

#%%

from graspologic.embed import mug2vec
from graspologic.utils import symmetrize

colors = sns.color_palette("Paired", desat=1)
side_names = [
    r"L $\rightarrow$ L",
    r"L $\rightarrow$ R",
    r"R $\rightarrow$ R",
    r"R $\rightarrow$ L",
]
graph_palette = {
    r"L $\rightarrow$ L": colors[1],
    r"L $\rightarrow$ R": colors[5],
    r"R $\rightarrow$ R": colors[0],
    r"R $\rightarrow$ L": colors[4],
}

graphs = [LL, LR, RR, RL]
graphs = [symmetrize(preprocess(A)) for A in graphs]
m2v = mug2vec()
graph_embeddings = m2v.fit_transform(graphs)
data = pd.DataFrame(data=graph_embeddings, columns=["0", "1"])
data["side_names"] = side_names
fig, ax = plt.subplots(1, 1, figsize=(4, 4))
sns.scatterplot(data=data, x="0", y="1", ax=ax, palette=graph_palette, hue="side_names")
ax.set(xlabel="CMDS-1", ylabel="CMDS-2", xticks=[], yticks=[])
ax.get_legend().remove()
ax.legend(bbox_to_anchor=(1, 1), loc="upper left", title="Side")
stashfig("cmds-ipsi-contra")

#%%

from src.visualization import CLASS_COLOR_DICT
from graspologic.plot import pairplot

left_labels = sided_meta["merge_class"].values[: len(lp_inds)]
palette = CLASS_COLOR_DICT

pg = pairplot(LL_embed_out[:, :4], labels=left_labels, palette=palette)
pg._legend.remove()

#%%

embeddings_by_side = {
    r"L $\rightarrow$ L": (LL_embed_out, LL_embed_in),
    r"L $\rightarrow$ R": (LR_embed_out, LR_embed_in),
    r"R $\rightarrow$ R": (RR_embed_out, RR_embed_in),
    r"R $\rightarrow$ L": (RL_embed_out, RL_embed_in),
}

for side_name, (out_embedding, in_embedding) in embeddings_by_side.items():
    Phat = out_embedding @ in_embedding.T
    adjplot(Phat, cbar=False)

#%%

from graspologic.align import OrthogonalProcrustes


def compute_procrustes_distance(X, Y, scale=False):
    X = X.copy()
    Y = Y.copy()
    if scale:
        X = X / np.linalg.norm(X)
        Y = Y / np.linalg.norm(Y)
    op = OrthogonalProcrustes()
    X_rot = op.fit_transform(X, Y)
    dist = np.linalg.norm(X_rot - Y, ord="fro")
    return dist


def compute_distances(X, Y):
    dists = []
    op_dist = compute_procrustes_distance(X, Y, scale=False)
    dists.append({"distance": op_dist, "distance_type": "op_dist"})
    scaled_op_dist = compute_procrustes_distance(X, Y, scale=True)
    dists.append({"distance": scaled_op_dist, "distance_type": "scaled_op_dist"})
    return dists


rows = []
for i, side_name1 in enumerate(side_names):
    for j, side_name2 in enumerate(side_names):
        out_embedding1, in_embedding1 = embeddings_by_side[side_name1]
        out_embedding2, in_embedding2 = embeddings_by_side[side_name2]
        out_dists = compute_distances(out_embedding1, out_embedding2)
        for dist in out_dists:
            dist["side1"] = side_name1
            dist["side2"] = side_name2
            dist["direction"] = "out"
            rows.append(dist)
        in_dists = compute_distances(in_embedding1, in_embedding2)
        for dist in in_dists:
            dist["side1"] = side_name1
            dist["side2"] = side_name2
            dist["direction"] = "in"
            rows.append(dist)

#%%
order = [
    r"L $\rightarrow$ L",
    r"R $\rightarrow$ R",
    r"L $\rightarrow$ R",
    r"R $\rightarrow$ L",
]
fig, axs = plt.subplots(2, 2, figsize=(10, 10))
results = pd.DataFrame(rows)
for i, direction in enumerate(["out", "in"]):
    for j, distance_type in enumerate(["op_dist", "scaled_op_dist"]):
        dists = results[
            (results["direction"] == direction)
            & (results["distance_type"] == distance_type)
        ]
        dists = dists.pivot(index="side1", columns="side2", values="distance")
        dists = dists.reindex(index=order, columns=order)
        ax = axs[i, j]
        sns.heatmap(dists, ax=ax, cmap="RdBu_r", center=0)
        plt.setp(ax.get_yticklabels(), rotation=0)
        plt.setp(ax.get_xticklabels(), rotation=45)
        ax.set(
            title=f"Distance = {distance_type}, direction = {direction}",
            xlabel="",
            ylabel="",
        )
plt.tight_layout()
stashfig("procrustes_dists")

#%%
