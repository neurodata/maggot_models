#%%
import os

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns

import pymaid
from graspologic.embed import AdjacencySpectralEmbed
from graspologic.utils import binarize, pass_to_ranks, remove_loops
from src.data import load_metagraph
from src.graph import MetaGraph
from src.io import savefig
from src.pymaid import start_instance
from src.utils import get_paired_inds
from src.visualization import set_theme

start_instance()


FNAME = os.path.basename(__file__)[:-3]
print(FNAME)


def stashfig(name, **kws):
    savefig(name, foldername=FNAME, save_on=True, print_out=False, **kws)


set_theme()

#%%

#%%


start_instance()

#%%

mg = load_metagraph("G")
mg = mg.make_lcc()
meta = mg.meta


def apply_annotation(meta, annotation, key):
    ids = pymaid.get_skids_by_annotation(annotation)
    ids = np.intersect1d(ids, meta.index)
    meta[key] = False
    meta.loc[ids, key] = True


apply_annotation(meta, "mw ipsilateral axon", "ipsilateral_axon")
apply_annotation(meta, "mw contralateral axon", "contralateral_axon")
apply_annotation(meta, "mw bilateral axon", "bilateral_axon")


#%%

n_components = 8
A = mg.adj.copy()
meta = mg.meta
meta["inds"] = range(len(meta))
A = pass_to_ranks(A)
ase = AdjacencySpectralEmbed(n_components=n_components)
X, Y = ase.fit_transform(A)

lp_inds, rp_inds = get_paired_inds(meta)
palette = dict(zip(range(len(lp_inds)), len(lp_inds) * ["black"]))

for x in range(n_components):
    for y in range(x + 1, n_components):
        data = pd.DataFrame(data=X[:, [x, y]], columns=["x", "y"])
        data["hue"] = meta["hemisphere"].values
        left_data = data.iloc[lp_inds].copy()
        right_data = data.iloc[rp_inds].copy()
        left_data["hue"] = range(len(left_data))
        right_data["hue"] = range(len(right_data))
        sided_data = pd.concat((left_data, right_data), axis=0)

        # plot without lines
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        sns.scatterplot(
            data=data, x="x", y="y", hue="hue", ax=ax, s=10, alpha=0.7, linewidth=0
        )
        ax.get_legend().remove()
        ax.legend(bbox_to_anchor=(1, 1), loc="upper left")
        ax.set(
            xticks=[],
            yticks=[],
            xlabel=f"Embedding dimension {x+1}",
            ylabel=f"Embedding dimension {y+1}",
        )
        stashfig(f"{x}-{y}-ase")

        # plot with lines
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        sns.scatterplot(
            data=data, x="x", y="y", hue="hue", ax=ax, s=10, alpha=0.7, linewidth=0
        )
        lines = sns.lineplot(
            data=sided_data,
            x="x",
            y="y",
            hue="hue",
            ax=ax,
            alpha=0.1,
            linewidth=1,
            palette=palette,
            legend=False,
            zorder=-1,
        )
        ax.get_legend().remove()
        ax.legend(bbox_to_anchor=(1, 1), loc="upper left")
        ax.set(
            xticks=[],
            yticks=[],
            xlabel=f"Embedding dimension {x+1}",
            ylabel=f"Embedding dimension {y+1}",
        )
        stashfig(f"{x}-{y}-ase-lines")

#%%
mg = load_metagraph("Gad")
mg = mg.make_lcc()
meta = mg.meta

apply_annotation(meta, "mw ipsilateral axon", "ipsilateral_axon")
apply_annotation(meta, "mw contralateral axon", "contralateral_axon")
apply_annotation(meta, "mw bilateral axon", "bilateral_axon")


meta["inds"] = range(len(meta))
lateral_meta = meta[
    (meta["hemisphere"].isin(["L", "R"]))
    & (meta["ipsilateral_axon"] | meta["bilateral_axon"] | meta["contralateral_axon"])
]
labels = lateral_meta[
    ["left", "ipsilateral_axon", "bilateral_axon", "contralateral_axon"]
].values
lateral_adj = mg.adj[np.ix_(lateral_meta.inds, lateral_meta.inds)].copy()
unique_rows = np.unique(labels, axis=0)
unique_rows = list(zip(*unique_rows.T))
labels = list(zip(*labels.T))
mapper = dict(zip(unique_rows, range(len(unique_rows))))
flat_labels = list(map(mapper.get, labels))
flat_labels
# flat_labels = np.vectorize(
#     labels,
# )
row_name_map = {
    (False, False, False, True): "Right contralateral",
    (False, False, True, False): "Right bilateral",
    (False, True, False, False): "Right ipsilateral",
    (True, False, False, True): "Left contralateral",
    (True, False, True, False): "Left bilateral",
    (True, True, False, False): "Left ipsilateral",
}
number_name_map = {
    0: "Right contralateral",
    1: "Right bilateral",
    2: "Right ipsilateral",
    3: "Left contralateral",
    4: "Left bilateral",
    5: "Left ipsilateral",
}

labels = [
    "Right contra",
    "Right bi",
    "Right ipsi",
    "Left contra",
    "Left bi",
    "Left ipsi",
]


from graspologic.models import SBMEstimator

sbme = SBMEstimator(directed=True, loops=True)
sbme.fit(binarize(lateral_adj), flat_labels)
fig, ax = plt.subplots(1, 1, figsize=(6, 6))
sns.heatmap(
    sbme.block_p_,
    square=True,
    annot=True,
    ax=ax,
    cmap="Purples",
    annot_kws=dict(fontsize=10),
    cbar=False,
    cbar_kws=dict(shrink=0.6),
)
ax.set(yticklabels=labels, xticklabels=labels)
ax.xaxis.tick_top()
plt.setp(ax.get_yticklabels(), rotation=0)
plt.setp(
    ax.get_xticklabels(), rotation=45, rotation_mode="anchor", ha="left", va="bottom"
)
stashfig("contra-bi-ipsi-sbm-model")