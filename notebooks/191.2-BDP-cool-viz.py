# %%
import os
from pathlib import Path

import colorcet as cc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymaid
import seaborn as sns
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA
from umap import UMAP

from src.data import load_metagraph
from src.io import savefig
from src.pymaid import start_instance
from src.visualization import CLASS_COLOR_DICT, set_theme

seed = 88888888
np.random.seed(seed)
set_theme()

FNAME = os.path.basename(__file__)[:-3]


def stashfig(name, **kws):
    savefig(name, foldername=FNAME, save_on=True, **kws)


mg = load_metagraph("G")

exp_dir = Path("maggot_models/experiments/matched_subgraph_omni_cluster/outs/")

# load full metadata with cluster labels
meta_loc = exp_dir / "meta-method=color_iso-d=8-bic_ratio=0.95-min_split=32.csv"
meta = pd.read_csv(meta_loc, index_col=0)

# load label tsv
embedding_label_loc = exp_dir / "omni_labels"
embedding_labels = pd.read_csv(embedding_label_loc, index_col=0, delimiter="\t")
embedding_labels.set_index("skid", inplace=True)
meta = meta.reindex(embedding_labels.index)

# load embedding tsv and reindex
embedding_loc = exp_dir / "omni_embed"
embedding = pd.read_csv(
    embedding_loc, index_col=None, header=None, delimiter="\t"
).values


mg = mg.reindex(meta.index, use_ids=True)
degrees = mg.calculate_degrees()
meta = pd.concat((meta, degrees), axis=1, ignore_index=False)
meta

# %% [markdown]
# ##
umapper = UMAP(
    n_neighbors=15, n_components=2, min_dist=1, metric="cosine", random_state=888888
)
y = meta["merge_class"].map(
    dict(zip(meta["merge_class"].unique(), range(meta["merge_class"].nunique())))
)
y = meta["lvl3_labels"].map(
    dict(zip(meta["lvl3_labels"].unique(), range(meta["lvl3_labels"].nunique())))
)
umap_embedding = umapper.fit_transform(embedding, y=y)
# move the embedding around a bit to make life easier later
mids = (umap_embedding.max(axis=0) + umap_embedding.min(axis=0)) / 2
umap_embedding -= mids
max_length = np.linalg.norm(umap_embedding, axis=1).max()
umap_embedding /= max_length

# %% [markdown]
# ##
columns = [f"umap_{i}" for i in range(umap_embedding.shape[1])]
plot_df = pd.DataFrame(data=umap_embedding, columns=columns, index=meta.index)
plot_df = pd.concat((plot_df, meta), axis=1, ignore_index=False)
plot_df

# %% [markdown]
# ##


def _segments_to_coords(x, segments, modifier=(1, 1, 1)):
    # REF: https://github.com/schlegelp/pymaid/blob/839863a523949ec64f1fa7112db558943ddc9adb/pymaid/plotting.py#L948
    """Turn lists of treenode_ids into coordinates.
    Parameters
    ----------
    x :         {pandas DataFrame, CatmaidNeuron}
                Must contain the nodes
    segments :  list of treenode IDs
    modifier :  ints, optional
                Use to modify/invert x/y/z axes.
    Returns
    -------
    coords :    list of tuples
                [ (x,y,z), (x,y,z ) ]
    """
    if not isinstance(modifier, np.ndarray):
        modifier = np.array(modifier)

    locs = {r.treenode_id: (r.x, r.y, r.z) for r in x.nodes.itertuples()}

    coords = [np.array([locs[tn] for tn in s]) * modifier for s in segments]

    return coords


def draw_neuron(skid, ax, neuron_scale=0.1, translation=[0, 0], color=None):
    skdata = pymaid.get_neuron(skid, with_connectors=True, with_abutting=True)
    neuron_points = skdata.nodes[["x", "y", "z"]].values

    pca = PCA(n_components=2)
    projected_points = pca.fit_transform(neuron_points)
    extents = projected_points.max(axis=0) - projected_points.min(axis=0)
    max_extent = extents.max()

    translation = np.asarray(translation)

    def _transform(points):
        points = pca.transform(points)
        points = points / max_extent
        points *= neuron_scale
        points = points + translation[None, :]

        return points

    coords = _segments_to_coords(skdata, skdata.segments)
    coords = [_transform(c) for c in coords]
    coords = np.vstack([np.append(t, [[None] * 2], axis=0) for t in coords])
    line = Line2D(coords[:, 0], coords[:, 1], lw=0.7, ls="-", color=color)
    ax.add_line(line)


# hue_key = "merge_class"
hue_key = "lvl3_labels"
cmap = "glasbey"
x_key = "umap_0"
y_key = "umap_1"
if hue_key == "merge_class":
    palette = CLASS_COLOR_DICT
else:
    if cmap == "husl":
        colors = sns.color_palette("husl", plot_df[hue_key].nunique())
    elif cmap == "glasbey":
        colors = cc.glasbey_light
    palette = dict(zip(plot_df[hue_key].unique(), colors))

# scatterplot for the nodes
fig, ax = plt.subplots(1, 1, figsize=(16, 16))
sns.scatterplot(
    data=plot_df,
    x=x_key,
    y=y_key,
    ax=ax,
    hue=hue_key,
    size="Total degree",
    palette=palette,
    sizes=(10, 80),
)
ax.axis("off")
ax.get_legend().remove()

# plot all of the edges
g = mg.g

rows = []
for i, (pre, post) in enumerate(g.edges):
    rows.append({"pre": pre, "post": post, "edge_idx": i})
edgelist = pd.DataFrame(rows)
edgelist["pre_class"] = edgelist["pre"].map(meta[hue_key])

pre_edgelist = edgelist.copy()
post_edgelist = edgelist.copy()

pre_edgelist["x"] = pre_edgelist["pre"].map(plot_df[x_key])
pre_edgelist["y"] = pre_edgelist["pre"].map(plot_df[y_key])

post_edgelist["x"] = post_edgelist["post"].map(plot_df[x_key])
post_edgelist["y"] = post_edgelist["post"].map(plot_df[y_key])

plot_edgelist = pd.concat((pre_edgelist, post_edgelist), axis=0, ignore_index=True)

edge_palette = dict(zip(edgelist["edge_idx"], edgelist["pre_class"].map(palette)))

pre_coords = list(zip(pre_edgelist["x"], pre_edgelist["y"]))
post_coords = list(zip(post_edgelist["x"], post_edgelist["y"]))
coords = list(zip(pre_coords, post_coords))
edge_colors = edgelist["pre_class"].map(palette)
lc = LineCollection(coords, colors=edge_colors, linewidths=0.15, alpha=0.15, zorder=0)
ax.add_collection(lc)

show_morphology = False
if show_morphology:
    start_instance()
    n_plot_neurons = 40
    neuron_scale = 0.18
    neuron_radius = 1.1
    angles = np.linspace(
        0, 2 * np.pi * (n_plot_neurons - 1) / n_plot_neurons, num=n_plot_neurons
    )
    xs = neuron_radius * np.cos(angles)
    ys = neuron_radius * np.sin(angles)
    vecs = np.stack((xs, ys), axis=1)

    neuron_angles = np.arctan2(plot_df["umap_1"], plot_df["umap_0"]).values
    neuron_angles[neuron_angles < 0] += 2 * np.pi

    random_angle_samples = np.random.normal(loc=angles, scale=0.025)  # was .02

    diffs = np.abs(random_angle_samples[:, None] - neuron_angles[None, :])
    min_diff_inds = np.argmin(diffs, axis=1)
    plot_neuron_ids = meta.index[min_diff_inds]

    for i, neuron_id in enumerate(plot_neuron_ids):
        color = palette[meta.loc[neuron_id, hue_key]]
        draw_neuron(
            neuron_id,
            ax,
            neuron_scale=neuron_scale,
            translation=vecs[i],
            color=color,
        )
    ax.relim()
    ax.autoscale_view()
stashfig(f"temp-maggot-brain-umap-omni-hue_key={hue_key}-seed={seed}")

#%%
from graspologic.utils import pass_to_ranks

adj = pass_to_ranks(mg.adj)
umap = UMAP(metric="precomputed")
adj_umap_embedding = umap.fit_transform(adj)
plot_df = pd.DataFrame(data=adj_umap_embedding, columns=columns, index=meta.index)
plot_df = pd.concat((plot_df, meta), axis=1, ignore_index=False)
plot_df


fig, ax = plt.subplots(1, 1, figsize=(16, 16))
sns.scatterplot(
    data=plot_df,
    x=x_key,
    y=y_key,
    ax=ax,
    hue=hue_key,
    size="Total degree",
    palette=palette,
    sizes=(10, 80),
)
ax.axis("off")
ax.get_legend().remove()

# %%


# %%
