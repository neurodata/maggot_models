#%% [markdown]
## Maggot embedding alignment
# > Investigating alignments of the maggot data embeddings
#
# - toc: false
# - badges: false
# - categories: [pedigo, graspologic, maggot]
# - hide: false
# - search_exclude: false
# %%
# collapse
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import StrMethodFormatter
from scipy.stats import ortho_group
from sklearn.neighbors import NearestNeighbors

import graspologic as gl
from graspologic.align import OrthogonalProcrustes, SeedlessProcrustes
from graspologic.embed import AdjacencySpectralEmbed, select_dimension
from graspologic.plot import pairplot
from graspologic.utils import augment_diagonal, pass_to_ranks
from src.data import load_metagraph
from src.graph import MetaGraph
from src.io import savefig
from src.visualization import CLASS_COLOR_DICT, adjplot, set_theme

print(f"graspologic version: {gl.__version__}")
print(f"seaborn version: {sns.__version__}")

set_theme()
palette = dict(zip(["Left", "Right", "OP", "SP"], sns.color_palette("Set1")))

FNAME = os.path.basename(__file__)[:-3]


def stashfig(name, **kws):
    savefig(name, foldername=FNAME, save_on=True, fmt="pdf", **kws)
    savefig(name, foldername=FNAME, save_on=True, fmt="png", dpi=300, **kws)


#%% [markdown]
### Load in the data, use only the known pairs
# %%
# collapse
mg = load_metagraph("G")
pair_meta = pd.read_csv(
    "maggot_models/experiments/graph_match/outs/pair_meta.csv", index_col=0
)
pair_key = "pair_id"
pair_meta = pair_meta[pair_meta[f"{pair_key[:-3]}"].isin(pair_meta.index)]
pair_meta = pair_meta.sort_values(["hemisphere", pair_key])
mg = mg.reindex(pair_meta.index.values, use_ids=True)
mg = MetaGraph(mg.adj, pair_meta)
n_pairs = len(pair_meta) // 2
left_inds = np.arange(n_pairs)
right_inds = left_inds.copy() + n_pairs
left_mg = MetaGraph(mg.adj[np.ix_(left_inds, left_inds)], mg.meta.iloc[left_inds])
right_mg = MetaGraph(mg.adj[np.ix_(right_inds, right_inds)], mg.meta.iloc[right_inds])
assert (left_mg.meta[pair_key].values == right_mg.meta[pair_key].values).all()

print(f"Working with {n_pairs} pairs.")

#%%[markdown]
### Plot the adjacency matrices, sorted by known pairs
# %%
# collapse
fig, axs = plt.subplots(1, 2, figsize=(10, 5))
adjplot(
    left_mg.adj,
    plot_type="scattermap",
    sizes=(1, 2),
    ax=axs[0],
    title=r"Left $\to$ left",
    color=palette["Left"],
)
adjplot(
    right_mg.adj,
    plot_type="scattermap",
    sizes=(1, 2),
    ax=axs[1],
    title=r"Right $\to$ right",
)
stashfig("left-right-adjs")

#%% [markdown]
### Embed the ipsilateral subgraphs
# - Computations are happening in `n_components` dimensions, where `n_components = 8` (dashed line in screeplot)
# - In many plots, I only show the first 4 dimensions just for clarity
#%%
# collapse


def plot_latents(left, right, title=""):
    plot_data = np.concatenate([left, right], axis=0)
    labels = np.array(["Left"] * len(left) + ["Right"] * len(right))
    pg = pairplot(plot_data[:, :4], labels=labels, title=title)
    return pg


def screeplot(sing_vals, elbow_inds, color=None, ax=None, label=None):
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(8, 4))
    plt.plot(range(1, len(sing_vals) + 1), sing_vals, color=color, label=label)
    plt.scatter(
        elbow_inds, sing_vals[elbow_inds - 1], marker="x", s=50, zorder=10, color=color
    )
    ax.set(ylabel="Singular value", xlabel="Index")
    return ax


def embed(adj, n_components=40):
    elbow_inds, elbow_vals = select_dimension(
        augment_diagonal(pass_to_ranks(adj)), n_elbows=4
    )
    elbow_inds = np.array(elbow_inds)
    ase = AdjacencySpectralEmbed(n_components=n_components)
    out_latent, in_latent = ase.fit_transform(pass_to_ranks(adj))
    return out_latent, in_latent, ase.singular_values_, elbow_inds


n_components = 8
max_n_components = 40
left_out_latent, left_in_latent, left_sing_vals, left_elbow_inds = embed(
    left_mg.adj, n_components=max_n_components
)
right_out_latent, right_in_latent, right_sing_vals, right_elbow_inds = embed(
    right_mg.adj, n_components=max_n_components
)

# plot the screeplot
fig, ax = plt.subplots(1, 1, figsize=(8, 4))
screeplot(left_sing_vals, left_elbow_inds, color=palette["Left"], ax=ax, label="Left")
screeplot(
    right_sing_vals, right_elbow_inds, color=palette["Right"], ax=ax, label="Right"
)
ax.legend()
ax.axvline(n_components, color="black", linewidth=1.5, linestyle="--")
stashfig("screeplot")

# plot the latent positions before any kind of alignment
plot_latents(
    left_out_latent, right_out_latent, title="Out latent positions (no alignment)"
)
stashfig("out-latent-no-align")
plot_latents(
    left_in_latent, right_in_latent, title="In latent positions (no alignment)"
)
stashfig("in-latent-no-align")

#%% [markdown]
### Align the embeddings
# Here I align first using Procrustes, and then using Seedless Procrustes initialized at
# the Procrustes solution. I will refer to this as "oracle initialization".
# %%
# collapse


def run_alignments(X, Y):
    op = OrthogonalProcrustes()
    X_trans_op = op.fit_transform(X, Y)
    sp = SeedlessProcrustes(init="custom", initial_Q=op.Q_)
    X_trans_sp = sp.fit_transform(X, Y)
    return X_trans_op, X_trans_sp


def calc_diff_norm(X, Y):
    return np.linalg.norm(X - Y, ord="fro")


op_left_out_latent, sp_left_out_latent = run_alignments(
    left_out_latent, right_out_latent
)
op_diff_norm = calc_diff_norm(op_left_out_latent, right_out_latent)
sp_diff_norm = calc_diff_norm(sp_left_out_latent, right_out_latent)

print(f"Procrustes diff. norm using true pairs: {op_diff_norm}")
print(f"Seedless Procrustes diff. norm using true pairs: {sp_diff_norm}")

#%%
# collapse

plot_latents(
    op_left_out_latent, right_out_latent, "Out latent positions (Procrustes alignment)"
)
stashfig("out-latent-procrustes")

plot_latents(
    sp_left_out_latent,
    right_out_latent,
    "Out latent positions (Seedless Procrustes alignment, oracle init)",
)
stashfig("out-latent-seedless-oracle")


# %%
# collapse

op_left_in_latent, sp_left_in_latent = run_alignments(left_in_latent, right_in_latent)

#%%[markdown]
### Looking at nearest neighbors in the aligned embeddings
# - I am concatenating the in and out embeddings *after* learning the alignments separately for each
# - "P by K" (short for precision by K) is the proportion of nodes for which their true pair is
# within its K nearest neighbors
# - I plot the cumulative density of the measure described above. So for a point `x, y` on the plot below,
# the result can be read as "`y` of all nodes have their true pair within their `x` nearest neighbors in
# the aligned embedded space".
# - I perform the experiment above using both orthogonal Procrustes (OP) and Seedless Procrustes (SP) with oracale initialization
# %%
# collapse


def calc_nn_ranks(target, query):
    n_pairs = len(target)
    nn = NearestNeighbors(n_neighbors=n_pairs, metric="euclidean")
    nn.fit(target)
    neigh_inds = nn.kneighbors(query, return_distance=False)
    true_neigh_inds = np.arange(n_pairs)
    _, ranks = np.where((neigh_inds == true_neigh_inds[:, None]))
    return ranks


def plot_rank_cdf(x, ax=None, color=None, label=None, max_rank=51):
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(8, 4))
    hist, _ = np.histogram(x, bins=np.arange(0, max_rank, 1))
    hist = hist / n_pairs
    ax.plot(np.arange(1, max_rank, 1), hist.cumsum(), color=color, label=label)
    ax.set(ylabel="Cumulative P by K", xlabel="K")
    return ax


op_left_composite_latent = np.concatenate(
    (op_left_out_latent, op_left_in_latent), axis=1
)
sp_left_composite_latent = np.concatenate(
    (sp_left_out_latent, sp_left_in_latent), axis=1
)
right_composite_latent = np.concatenate((right_out_latent, right_in_latent), axis=1)


op_ranks = calc_nn_ranks(op_left_composite_latent, right_composite_latent)
sp_ranks = calc_nn_ranks(sp_left_composite_latent, right_composite_latent)

fig, ax = plt.subplots(1, 1, figsize=(8, 4))
plot_rank_cdf(op_ranks, color=palette["OP"], ax=ax, label="OP")
plot_rank_cdf(sp_ranks, color=palette["SP"], ax=ax, label="SP")
ax.legend()
stashfig("rank-cdf")

#%%
right_meta = right_mg.meta.copy()
right_meta["pair_rank"] = sp_ranks
right_meta["reciprocal_rank"] = 1 / (sp_ranks + 1)

mean_ranks = right_meta.groupby("merge_class")["pair_rank"].mean()
sizes = right_meta.groupby("merge_class").size()
mean_ranks = pd.concat((mean_ranks, sizes), axis=1)

cluster_meta = pd.read_csv(
    "maggot_models/experiments/matched_subgraph_omni_cluster/outs/meta-method=color_iso-d=8-bic_ratio=0.95-min_split=32.csv",
    index_col=0,
)
level = 4
label_key = f"lvl{level}_labels"
right_meta[label_key] = cluster_meta[label_key]
right_meta.groupby(label_key)["reciprocal_rank"].mean()
right_meta["merge_class"] = cluster_meta["simple_class"]

class_order = [
    "Sens",
    "LN",
    "PN",
    "KC",
    "LHN",
    "MBIN",
    "Sens2o",
    "unk",
    "MBON",
    "FBN",
    "CN",
    "PreO",
    "Outs",
    "Motr",
]

mrr_by_group = right_meta.groupby(label_key)["reciprocal_rank"].mean()
mrr_by_group.sort_values(inplace=True, ascending=False)
group_counts = right_meta.groupby([label_key, "merge_class"]).size()
group_counts = group_counts.reindex(mrr_by_group.index, level=0)
group_counts = group_counts.reindex(class_order[::-1], level=1)


width = 0.5

fig, axs = plt.subplots(
    2, 1, figsize=(16, 8), gridspec_kw=dict(hspace=0.01, height_ratios=[0.7, 0.3])
)
ax = axs[1]
ax.set(
    xlim=(-1, len(group_counts.index.unique(0))),
    ylim=(0, 1),
    xlabel=f"Cluster (level = {level})",
    xticks=[],
    yticks=[],
)

for i, cluster_name in enumerate(group_counts.index.unique(0)):
    cluster_counts = group_counts.loc[cluster_name]
    cluster_props = cluster_counts / cluster_counts.sum()
    start = 0
    for class_name, prop in cluster_props.iteritems():
        ax.bar(
            x=i,
            height=prop,
            width=width,
            bottom=start,
            color=CLASS_COLOR_DICT[class_name],
        )
        start += prop

ax.spines["left"].set_visible(False)
ax.spines["bottom"].set_visible(False)

ax = axs[0]
pad = 0.05

ax.set(
    xlim=(-1, len(group_counts.index.unique(0))),
    ylim=(mrr_by_group.min() - 3 * pad, mrr_by_group.max() + 2 * pad),
    ylabel="Pair mean reciprocal rank",
    xticks=[],
    yticks=[mrr_by_group.min(), mrr_by_group.max()],
)
ax.yaxis.set_major_formatter(StrMethodFormatter("{x:.2f}"))

for i, cluster_name in enumerate(group_counts.index.unique(0)):
    mrr = mrr_by_group.loc[cluster_name]
    ax.plot([i], [mrr], marker="o", color="black", markersize=8)

ax.text(0, mrr_by_group.max() + pad, "Most bilateral symmetry")
ax.text(
    len(mrr_by_group),
    mrr_by_group.min() - pad,
    "Least bilateral symmetry",
    ha="right",
    va="top",
)
stashfig(f"mrr-by-cluster-lvl={level}")
