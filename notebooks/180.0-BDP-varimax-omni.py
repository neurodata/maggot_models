# %% [markdown]
# ##
import os
import time

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import poisson

from graspy.embed import OmnibusEmbed, selectSVD
from graspy.match import GraphMatch
from graspy.models import DCSBMEstimator, SBMEstimator
from graspy.utils import (
    augment_diagonal,
    binarize,
    pass_to_ranks,
    remove_loops,
    to_laplace,
)
from src.cluster import BinaryCluster
from src.data import load_metagraph
from src.graph import MetaGraph
from src.hierarchy import signal_flow
from src.io import savecsv, savefig
from src.utils import get_paired_inds
from src.visualization import (
    CLASS_COLOR_DICT,
    add_connections,
    adjplot,
    plot_color_labels,
    plot_double_dendrogram,
    plot_single_dendrogram,
)

# For saving outputs
FNAME = os.path.basename(__file__)[:-3]
print(FNAME)

# plotting settings
rc_dict = {
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.formatter.limits": (-3, 3),
    "figure.figsize": (6, 3),
    "figure.dpi": 100,
    "axes.edgecolor": "lightgrey",
    "ytick.color": "grey",
    "xtick.color": "grey",
    "axes.labelcolor": "dimgrey",
    "text.color": "dimgrey",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial"],
}
for key, val in rc_dict.items():
    mpl.rcParams[key] = val
context = sns.plotting_context(context="talk", font_scale=1, rc=rc_dict)
sns.set_context(context)

np.random.seed(8888)


def stashfig(name, **kws):
    savefig(name, foldername=FNAME, save_on=True, **kws)


def stashcsv(df, name, **kws):
    savecsv(df, name, foldername=FNAME, **kws)


def plot_pairs(
    X, labels, model=None, left_pair_inds=None, right_pair_inds=None, equal=False
):
    """ Plots pairwise dimensional projections, and draws lines between known pair neurons

    Parameters
    ----------
    X : [type]
        [description]
    labels : [type]
        [description]
    model : [type], optional
        [description], by default None
    left_pair_inds : [type], optional
        [description], by default None
    right_pair_inds : [type], optional
        [description], by default None
    equal : bool, optional
        [description], by default False

    Returns
    -------
    [type]
        [description]
    """

    n_dims = X.shape[1]

    fig, axs = plt.subplots(
        n_dims, n_dims, sharex=False, sharey=False, figsize=(20, 20)
    )
    data = pd.DataFrame(data=X)
    data["label"] = labels

    for i in range(n_dims):
        for j in range(n_dims):
            ax = axs[i, j]
            ax.axis("off")
            if i < j:
                sns.scatterplot(
                    data=data,
                    x=j,
                    y=i,
                    ax=ax,
                    alpha=0.7,
                    linewidth=0,
                    s=8,
                    legend=False,
                    hue="label",
                    palette=CLASS_COLOR_DICT,
                )
                if left_pair_inds is not None and right_pair_inds is not None:
                    add_connections(
                        data.iloc[left_pair_inds, j],
                        data.iloc[right_pair_inds, j],
                        data.iloc[left_pair_inds, i],
                        data.iloc[right_pair_inds, i],
                        ax=ax,
                    )

    plt.tight_layout()
    return fig, axs


def preprocess_adjs(adjs, method="ase"):
    """Preprocessing necessary prior to embedding a graph, opetates on a list

    Parameters
    ----------
    adjs : list of adjacency matrices
        [description]
    method : str, optional
        [description], by default "ase"

    Returns
    -------
    [type]
        [description]
    """
    adjs = [pass_to_ranks(a) for a in adjs]
    adjs = [a + 1 / a.size for a in adjs]
    if method == "ase":
        adjs = [augment_diagonal(a) for a in adjs]
    elif method == "lse":  # haven't really used much. a few params to look at here
        adjs = [to_laplace(a) for a in adjs]
    return adjs


def omni(
    adjs,
    n_components=4,
    remove_first=None,
    concat_graphs=True,
    concat_directed=True,
    method="ase",
):
    """Omni with a few extra (optional) bells and whistles for concatenation post embed

    Parameters
    ----------
    adjs : [type]
        [description]
    n_components : int, optional
        [description], by default 4
    remove_first : [type], optional
        [description], by default None
    concat_graphs : bool, optional
        [description], by default True
    concat_directed : bool, optional
        [description], by default True
    method : str, optional
        [description], by default "ase"

    Returns
    -------
    [type]
        [description]
    """
    adjs = preprocess_adjs(adjs, method=method)
    omni = OmnibusEmbed(n_components=n_components, check_lcc=False, n_iter=10)
    embed = omni.fit_transform(adjs)
    if concat_directed:
        embed = np.concatenate(
            embed, axis=-1
        )  # this is for left/right latent positions
    if remove_first is not None:
        embed = embed[remove_first:]
    if concat_graphs:
        embed = np.concatenate(embed, axis=0)
    return embed


def sort_mg(mg, level_names):
    """Required sorting prior to plotting the dendrograms

    Parameters
    ----------
    mg : [type]
        [description]
    level_names : [type]
        [description]

    Returns
    -------
    [type]
        [description]
    """
    meta = mg.meta
    sort_class = level_names + ["merge_class"]
    class_order = ["sf"]
    total_sort_by = []
    for sc in sort_class:
        for co in class_order:
            class_value = meta.groupby(sc)[co].mean()
            meta[f"{sc}_{co}_order"] = meta[sc].map(class_value)
            total_sort_by.append(f"{sc}_{co}_order")
        total_sort_by.append(sc)
    mg = mg.sort_values(total_sort_by, ascending=False)
    return mg


def plot_adjacencies(full_mg, axs, lowest_level=7):
    level_names = [f"lvl{i}_labels" for i in range(lowest_level + 1)]
    pal = sns.color_palette("deep", 1)
    model = DCSBMEstimator
    for level in np.arange(lowest_level + 1):
        ax = axs[0, level]
        adj = binarize(full_mg.adj)
        _, _, top, _ = adjplot(
            adj,
            ax=ax,
            plot_type="scattermap",
            sizes=(0.5, 0.5),
            sort_class=["hemisphere"] + level_names[: level + 1],
            item_order=["merge_class_sf_order", "merge_class", "sf"],
            class_order="sf",
            meta=full_mg.meta,
            palette=CLASS_COLOR_DICT,
            colors="merge_class",
            ticks=False,
            gridline_kws=dict(linewidth=0.2, color="grey", linestyle="--"),
            color=pal[0],
        )
        top.set_title(f"Level {level} - Data")

        labels = full_mg.meta[f"lvl{level}_labels_side"]
        estimator = model(directed=True, loops=True)
        uni_labels, inv = np.unique(labels, return_inverse=True)
        estimator.fit(adj, inv)
        sample_adj = np.squeeze(estimator.sample())
        ax = axs[1, level]
        _, _, top, _ = adjplot(
            sample_adj,
            ax=ax,
            plot_type="scattermap",
            sizes=(0.5, 0.5),
            sort_class=["hemisphere"] + level_names[: level + 1],
            item_order=["merge_class_sf_order", "merge_class", "sf"],
            class_order="sf",
            meta=full_mg.meta,
            palette=CLASS_COLOR_DICT,
            colors="merge_class",
            ticks=False,
            gridline_kws=dict(linewidth=0.2, color="grey", linestyle="--"),
            color=pal[0],
        )
        top.set_title(f"Level {level} - DCSBM sample")


def plot_model_liks(adj, meta, lp_inds, rp_inds, ax, n_levels=10, model_name="DCSBM"):
    plot_df = calc_model_liks(adj, meta, lp_inds, rp_inds, n_levels=n_levels)
    sns.lineplot(
        data=plot_df[plot_df["model"] == model_name],
        hue="test",
        x="level",
        y="norm_score",
        style="train_side",
        markers=True,
    )
    ax.set_ylabel(f"{model_name} normalized log lik.")
    ax.set_xlabel("Level")


def plot_pairedness(meta, lp_inds, rp_inds, ax, n_levels=10, n_shuffles=10):
    rows = []
    for l in range(n_levels + 1):
        pred_labels = meta[f"lvl{l}_labels"].values.copy()
        p_same = calc_pairedness(pred_labels, lp_inds, rp_inds)
        rows.append(dict(p_same_cluster=p_same, labels="True", level=l))
        # look at random chance
        for i in range(n_shuffles):
            np.random.shuffle(pred_labels)
            p_same = calc_pairedness(pred_labels, lp_inds, rp_inds)
            rows.append(dict(p_same_cluster=p_same, labels="Shuffled", level=l))
    plot_df = pd.DataFrame(rows)

    sns.lineplot(
        data=plot_df,
        x="level",
        y="p_same_cluster",
        ax=ax,
        hue="labels",
        markers=True,
        style="labels",
    )
    ax.set_ylabel("P same cluster")
    ax.set_xlabel("Level")


def calc_pairedness(pred_labels, lp_inds, rp_inds):
    left_labels = pred_labels[lp_inds]
    right_labels = pred_labels[rp_inds]
    n_same = (left_labels == right_labels).sum()
    p_same = n_same / len(lp_inds)
    return p_same


def plot_n_clusters(meta, ax, n_levels=10):
    n_clusters = []
    for l in range(n_levels + 1):
        n_clusters.append(meta[f"lvl{l}_labels"].nunique())
    sns.lineplot(x=range(n_levels + 1), y=n_clusters, ax=ax)
    sns.scatterplot(x=range(n_levels + 1), y=n_clusters, ax=ax)
    ax.set_ylabel("Clusters per side")
    ax.set_xlabel("Level")


def plot_cluster_size(meta, ax, n_levels=10, side=True, boxes=False):
    fontsize = mpl.rcParams["font.size"]
    group_key = "lvl{}_labels"
    if side:
        group_key += "_side"
    size_dfs = []
    for l in range(n_levels + 1):
        sizes = meta.groupby(group_key.format(l)).size().values
        sizes = pd.DataFrame(data=sizes, columns=["Size"])
        sizes["Level"] = l
        size_dfs.append(sizes)
        min_size = sizes["Size"].min()
        max_size = sizes["Size"].max()
        ax.annotate(
            s=max_size,
            xy=(l, max_size),
            xytext=(0, 30),
            ha="center",
            va="bottom",
            textcoords="offset pixels",
            fontsize=0.75 * fontsize,
        )
        ax.annotate(
            s=min_size,
            xy=(l, min_size),
            xytext=(0, -30),
            ha="center",
            va="top",
            textcoords="offset pixels",
            fontsize=0.75 * fontsize,
        )

    size_df = pd.concat(size_dfs)
    if boxes:
        sns.boxenplot(data=size_df, x="Level", y="Size", ax=ax)
    sns.stripplot(
        data=size_df,
        x="Level",
        y="Size",
        ax=ax,
        alpha=0.5,
        s=4,
        jitter=0.3,
        color="grey",
    )
    ax.set_yscale("log")
    ylim = ax.get_ylim()
    ax.set_ylim((0.4, ylim[1]))
    if side:
        ax.set_ylabel("Size per side")
    ax.set_xlabel("Level")


def calc_model_liks(adj, meta, lp_inds, rp_inds, n_levels=10):
    rows = []
    for l in range(n_levels + 1):
        labels = meta[f"lvl{l}_labels"].values
        left_adj = binarize(adj[np.ix_(lp_inds, lp_inds)])
        left_adj = remove_loops(left_adj)
        right_adj = binarize(adj[np.ix_(rp_inds, rp_inds)])
        right_adj = remove_loops(right_adj)
        for model, name in zip([DCSBMEstimator, SBMEstimator], ["DCSBM", "SBM"]):
            estimator = model(directed=True, loops=False)
            uni_labels, inv = np.unique(labels, return_inverse=True)
            estimator.fit(left_adj, inv[lp_inds])
            train_left_p = estimator.p_mat_
            train_left_p[train_left_p == 0] = 1 / train_left_p.size

            n_params = estimator._n_parameters() + len(uni_labels)

            score = poisson.logpmf(left_adj, train_left_p).sum()
            rows.append(
                dict(
                    train_side="Left",
                    test="Same",
                    test_side="Left",
                    score=score,
                    level=l,
                    model=name,
                    n_params=n_params,
                    norm_score=score / left_adj.sum(),
                )
            )
            score = poisson.logpmf(right_adj, train_left_p).sum()
            rows.append(
                dict(
                    train_side="Left",
                    test="Opposite",
                    test_side="Right",
                    score=score,
                    level=l,
                    model=name,
                    n_params=n_params,
                    norm_score=score / right_adj.sum(),
                )
            )

            estimator = model(directed=True, loops=False)
            estimator.fit(right_adj, inv[rp_inds])
            train_right_p = estimator.p_mat_
            train_right_p[train_right_p == 0] = 1 / train_right_p.size

            n_params = estimator._n_parameters() + len(uni_labels)

            score = poisson.logpmf(left_adj, train_right_p).sum()
            rows.append(
                dict(
                    train_side="Right",
                    test="Opposite",
                    test_side="Left",
                    score=score,
                    level=l,
                    model=name,
                    n_params=n_params,
                    norm_score=score / left_adj.sum(),
                )
            )
            score = poisson.logpmf(right_adj, train_right_p).sum()
            rows.append(
                dict(
                    train_side="Right",
                    test="Same",
                    test_side="Right",
                    score=score,
                    level=l,
                    model=name,
                    n_params=n_params,
                    norm_score=score / right_adj.sum(),
                )
            )
    return pd.DataFrame(rows)


def plot_clustering_results(adj, meta, basename, lowest_level=7):
    level_names = [f"lvl{i}_labels" for i in range(lowest_level + 1)]

    mg = MetaGraph(adj, meta)
    mg = sort_mg(mg, level_names)
    meta = mg.meta
    adj = mg.adj

    # set up figure
    # analysis, bars, colors, graph graph graph...
    n_col = 1 + 2 + 1 + lowest_level + 1
    n_row = 6
    width_ratios = 4 * [1] + (lowest_level + 1) * [1.5]

    fig = plt.figure(
        constrained_layout=False, figsize=(5 * 4 + (lowest_level + 1) * 8.5, 20)
    )
    gs = plt.GridSpec(nrows=n_row, ncols=n_col, figure=fig, width_ratios=width_ratios)

    # plot the dendrograms
    dend_axs = []
    dend_axs.append(fig.add_subplot(gs[:, 1]))  # left
    dend_axs.append(fig.add_subplot(gs[:, 2]))  # right
    dend_axs.append(fig.add_subplot(gs[:, 3]))  # colormap
    plot_double_dendrogram(meta, dend_axs[:-1], lowest_level=lowest_level)
    plot_color_labels(meta, dend_axs[-1])

    # plot the adjacency matrices for data and sampled data
    adj_axs = np.empty((2, lowest_level + 1), dtype="O")
    offset = 4
    for level in np.arange(lowest_level + 1):
        ax = fig.add_subplot(gs[: n_row // 2, level + offset])
        adj_axs[0, level] = ax
        ax = fig.add_subplot(gs[n_row // 2 :, level + offset])
        adj_axs[1, level] = ax
    plot_adjacencies(mg, adj_axs, lowest_level=lowest_level)

    mg = mg.sort_values(["hemisphere", "pair_id"], ascending=True)
    meta = mg.meta
    adj = mg.adj
    n_pairs = len(meta) // 2
    lp_inds = np.arange(n_pairs)
    rp_inds = np.arange(n_pairs) + n_pairs
    n_levels = 9  # how many levels to show in the curve plots

    # plot the pairedness in the top left
    palette = sns.color_palette("deep", 2)
    sns.set_palette(palette)
    ax = fig.add_subplot(gs[:2, 0])
    plot_pairedness(meta, lp_inds, rp_inds, ax, n_levels=n_levels)

    # plot the likelihood curves in the middle left
    palette = sns.color_palette("deep")
    palette = [palette[2], palette[4]]  # green, purple,
    sns.set_palette(palette)
    ax = fig.add_subplot(gs[2:4, 0], sharex=ax)
    plot_model_liks(adj, meta, lp_inds, rp_inds, ax, n_levels=n_levels)

    # plot the number of clusters in the bottom left
    palette = sns.color_palette("deep")
    palette = [palette[5]]  # brown
    sns.set_palette(palette)
    ax = fig.add_subplot(gs[4:6, 0], sharex=ax)
    plot_cluster_size(meta, ax, n_levels=n_levels)

    # finish up
    plt.tight_layout()
    stashfig(f"megafig-lowest={lowest_level}" + basename)
    plt.close()

    # make a single barplot for each level
    mg = sort_mg(mg, level_names)
    meta = mg.meta
    adj = mg.adj
    width_ratios = [0.7, 0.3]

    for l in range(lowest_level + 1):
        fig = plt.figure(figsize=(10, 30))
        fig, axs = plt.subplots(
            1, 2, figsize=(10, 30), gridspec_kw=dict(width_ratios=width_ratios)
        )
        plot_single_dendrogram(meta, axs[0], draw_labels=True, lowest_level=l)
        plot_color_labels(meta, axs[1])
        axs[1].yaxis.tick_right()
        stashfig(f"bars-lowest={l}" + basename)
        plt.close()


# %% [markdown]
# ## Load and preprocess data
graph_type = "G"
VERSION = "2020-06-10"
master_mg = load_metagraph(graph_type, version=VERSION)
mg = MetaGraph(master_mg.adj, master_mg.meta)
mg = mg.remove_pdiff()
meta = mg.meta.copy()

# remove low degree neurons
degrees = mg.calculate_degrees()
quant_val = np.quantile(degrees["Total edgesum"], 0.05)
idx = meta[degrees["Total edgesum"] > quant_val].index
print(quant_val)
mg = mg.reindex(idx, use_ids=True)
mg = mg.make_lcc()
meta = mg.meta

# TODO the following block needs to be cleaned up, should make this easy to do with
# MetaGraph class
temp_meta = meta[meta["left"] | meta["right"]]
unpair_idx = temp_meta[~temp_meta["pair"].isin(temp_meta.index)].index
meta.loc[unpair_idx, ["pair", "pair_id"]] = -1

left_idx = meta[meta["left"]].index
left_mg = MetaGraph(mg.adj, mg.meta)
left_mg = left_mg.reindex(left_idx, use_ids=True)
left_mg = left_mg.sort_values(["pair_id"], ascending=False)
print(len(left_mg))
right_idx = meta[meta["right"]].index
right_mg = MetaGraph(mg.adj, mg.meta)
right_mg = right_mg.reindex(right_idx, use_ids=True)
right_mg = right_mg.sort_values(["pair_id"], ascending=False)
right_mg = right_mg.reindex(right_mg.meta.index[: len(left_mg)], use_ids=True)
print(len(right_mg))

assert (right_mg.meta["pair_id"].values == left_mg.meta["pair_id"].values).all()

# %% [markdown]
# ## Pair the unpaired neurons using graph matching

n_pairs = len(right_mg.meta[right_mg.meta["pair_id"] != -1])
left_adj = left_mg.adj
right_adj = right_mg.adj
left_seeds = right_seeds = np.arange(n_pairs)
currtime = time.time()
gm = GraphMatch(n_init=50, init_method="rand", eps=1.0, shuffle_input=False)
gm.fit(left_adj, right_adj, seeds_A=left_seeds, seeds_B=right_seeds)
print(f"{(time.time() - currtime)/60:0.2f} minutes elapsed for graph matching")

fig, axs = plt.subplots(1, 2, figsize=(20, 10))
perm_inds = gm.perm_inds_
true_pairs = n_pairs * ["paired"] + (len(left_adj) - n_pairs) * ["unpaired"]
pal = sns.color_palette("deep", 3)
color_dict = dict(zip(["unpaired", "paired"], pal[1:]))
_, _, top, _ = adjplot(
    left_adj,
    ax=axs[0],
    plot_type="scattermap",
    sizes=(1, 1),
    colors=true_pairs,
    palette=color_dict,
    color=pal[0],
    ticks=True,
)
top.set_title("Left")
_, _, top, _ = adjplot(
    right_adj[np.ix_(perm_inds, perm_inds)],
    ax=axs[1],
    plot_type="scattermap",
    sizes=(1, 1),
    colors=true_pairs,
    palette=color_dict,
    color=pal[0],
    ticks=True,
)
top.set_title("Right")
plt.tight_layout()
stashfig("gm-results-adj")

# %% [markdown]
# ## Apply the pairs

left_pair_id = pd.Series(index=left_mg.meta.index, data=np.arange(len(left_mg)))
right_pair_idx = right_mg.meta.index[perm_inds]
right_pair_id = pd.Series(index=right_pair_idx, data=np.arange(len(right_mg)))
pair_id = pd.concat((left_pair_id, right_pair_id))
mg = load_metagraph(graph_type, version=VERSION)
mg = mg.reindex(pair_id.index, use_ids=True)
mg.meta["pair_id"] = pair_id


meta = mg.meta
meta["inds"] = range(len(meta))
adj = mg.adj.copy()
# inds on left and right that correspond to the bilateral pairs
lp_inds, rp_inds = get_paired_inds(meta, check_in=False)
left_inds = meta[meta["left"]]["inds"]

print(f"Neurons left after preprocessing: {len(mg)}")


# %% [markdown]
# ## Load the 4-color graphs
# convention is "ad" = axon -> dendrite and so on...
graph_types = ["Gad", "Gaa", "Gdd", "Gda"]  # "Gs"]
adjs = []
for g in graph_types:
    temp_mg = load_metagraph(g, version=VERSION)
    # this line is important, to make the graphs aligned
    temp_mg.reindex(mg.meta.index, use_ids=True)
    temp_adj = temp_mg.adj
    adjs.append(temp_adj)

# %% [markdown]
# ## SVDer
# 8, 16 seems to work
n_omni_components = 8  # this is used for all of the embedings initially
n_svd_components = 16  # this is for the last step
method = "ase"  # one could also do LSE


def svd(X, n_components=n_svd_components):
    return selectSVD(X, n_components=n_components, algorithm="full")[0]


# %% [markdown]
# # Run Omni and deal with latent positions appropriately

omni_method = "color_iso"
currtime = time.time()

if omni_method == "iso":
    full_adjs = [
        adj[np.ix_(lp_inds, lp_inds)],
        adj[np.ix_(lp_inds, rp_inds)],
        adj[np.ix_(rp_inds, rp_inds)],
        adj[np.ix_(rp_inds, lp_inds)],
    ]
    out_embed, in_embed = omni(
        full_adjs,
        n_components=n_omni_components,
        remove_first=None,
        concat_graphs=False,
        concat_directed=False,
        method=method,
    )

    # this step is weird to explain - for the omnibus embedding I've jointly embedded
    # the left ipsilateral, left-to-right contralateral, right ipsilateral, and
    # right-to-left contralateral subgraphs (so taken together, it's everything).
    # ipsi out, contra out, ipsi in, contra in
    left_embed = np.concatenate(
        (out_embed[0], out_embed[1], in_embed[0], in_embed[3]), axis=1
    )
    right_embed = np.concatenate(
        (out_embed[2], out_embed[3], in_embed[2], in_embed[1]), axis=1
    )
    omni_iso_embed = np.concatenate((left_embed, right_embed), axis=0)

    svd_embed = svd(omni_iso_embed)

elif omni_method == "color_iso":
    # break up all 4 color graphs by left/right contra/ipsi as described above
    all_sub_adjs = []
    for a in adjs:
        sub_adjs = [
            a[np.ix_(lp_inds, lp_inds)],
            a[np.ix_(lp_inds, rp_inds)],
            a[np.ix_(rp_inds, rp_inds)],
            a[np.ix_(rp_inds, lp_inds)],
        ]
        all_sub_adjs += sub_adjs

    # embed all of them jointly using omni
    out_embed, in_embed = omni(
        all_sub_adjs,
        n_components=n_omni_components,
        remove_first=None,
        concat_graphs=False,
        concat_directed=False,
        method=method,
    )

    # again, this part is tricky. have to get the right embeddings corresponding to the
    # correct colors/graphs/directions
    # note that for the left side I am grabbing the in and out latent positions for the
    # left-left subgraph, the out latent positions for the left-to-right subgraph, and
    # the in latent positions for the right-to-left subgraph. This code just does that
    # for each color and then concatenates them all together
    color_embeds = []
    for i in range(len(adjs)):
        start = i * 4  # 4 is for contra/ipsi left/right
        left_embed = np.concatenate(
            (
                out_embed[0 + start],
                out_embed[1 + start],
                in_embed[0 + start],
                in_embed[3 + start],
            ),
            axis=1,
        )
        right_embed = np.concatenate(
            (
                out_embed[2 + start],
                out_embed[3 + start],
                in_embed[2 + start],
                in_embed[1 + start],
            ),
            axis=1,
        )
        color_embed = np.concatenate((left_embed, right_embed), axis=0)
        color_embeds.append(color_embed)

    omni_color_embed = np.concatenate(color_embeds, axis=1)
    # after concatenating, SVD them all down to a lower dimension again
    # this step may be suspect...
    # but don't want to run GMM in very high d
    svd_embed = svd(omni_color_embed)

print(f"{(time.time() - currtime)/60:0.2f} minutes elapsed for embedding")
#%%

from factor_analyzer.rotator import Rotator
from graspy.plot import pairplot

curr_time = time.time()
rot = Rotator()
rot_embed = rot.fit_transform(svd_embed)
print(f"{(time.time() - curr_time) / 60} minutes elapsed (wall time).")
# %%

pg = pairplot(rot_embed, labels=meta["merge_class"].values, palette=CLASS_COLOR_DICT)
pg._legend.remove()
stashfig("varimax-embedding")

# %%

curr_time = time.time()
rot = Rotator()
rot_embed = rot.fit_transform(svd_embed)
print(f"{(time.time() - curr_time) / 60} minutes elapsed (wall time).")
plot_log = True
for i, dim in enumerate(rot_embed.T):
    sort_inds = np.argsort(dim)
    plot_df = pd.DataFrame()
    if plot_log:
        dim = np.log(np.abs(dim))
    plot_df["dim"] = dim[sort_inds]
    plot_df["labels"] = meta["merge_class"].values[sort_inds]
    plot_df["ind"] = range(len(plot_df))
    fig, axs = plt.subplots(
        1, 2, figsize=(10, 4), gridspec_kw=dict(width_ratios=[2, 1]), sharey=True
    )
    ax = axs[0]
    sns.scatterplot(
        x="ind",
        y="dim",
        hue="labels",
        data=plot_df,
        ax=ax,
        palette=CLASS_COLOR_DICT,
        legend=False,
        s=5,
        alpha=0.8,
        linewidth=0,
    )
    ax.set(
        xticks=[], yticks=[], ylabel=f"Component {i}", xlabel="Index (sorted)",
    )
    if not plot_log:
        ax.axhline(0, color="grey", linewidth=1, linestyle="--")

    is_zero = np.isclose(dim, 0, atol=1e-5)
    print(is_zero.mean())

    ax = axs[1]
    sns.distplot(dim, ax=ax, vertical=True)
    ax.set(xticks=[], yticks=[], ylabel="", xlabel="Frequency")
    stashfig(f"component{i}-varimax-plot_log={plot_log}")
    # ax.set_xscale("log")

# %% [markdown]
# ##

n_pairs = len(lp_inds)
new_lp_inds = np.arange(n_pairs)
new_rp_inds = np.arange(n_pairs) + n_pairs
new_meta = meta.iloc[np.concatenate((lp_inds, rp_inds), axis=0)].copy()
labels = new_meta["merge_class"].values

columns = [
    "hemisphere",
    "class1",
    "class2",
    "merge_class",
    "pair",
    "pair_id",
    "lineage",
    "name",
]
embedding_out = f"maggot_models/notebooks/outs/{FNAME}/tsvs/omni_"

from topologic.io import tensor_projection_writer

save_meta = new_meta[columns].copy()
save_meta.index.name = "skid"
save_meta = save_meta.reset_index()
# save_meta.set_index("name")

tensor_projection_writer(
    embedding_out + "embed",
    embedding_out + "labels",
    svd_embed,
    save_meta.values.tolist(),
)
save_meta.to_csv(embedding_out + "labels", sep="\t")


plot_pairs(
    svd_embed[:, :8], labels, left_pair_inds=new_lp_inds, right_pair_inds=new_rp_inds
)
stashfig(f"pairs-method={omni_method}")

# %% [markdown]
# ## Clustering

# parameters
n_levels = 10  # max # of splits in the recursive clustering
metric = "bic"  # metric on which to decide best split
# bic_ratio = 1  # ratio used for whether or not to split
# d = 8  # embedding dimension

params = [
    # {"d": 6, "bic_ratio": 0.8, "min_split": 32},
    # {"d": 6, "bic_ratio": 0.9, "min_split": 32},
    # {"d": 8, "bic_ratio": 0.8, "min_split": 32},
    # {"d": 8, "bic_ratio": 0.9, "min_split": 32},
    # {"d": 8, "bic_ratio": 0.95, "min_split": 32},
    {"d": 8, "bic_ratio": 1, "min_split": 32},
    # {"d": 10, "bic_ratio": 0.9, "min_split": 32},
]

for p in params:
    print(p)
    d = p["d"]
    bic_ratio = p["bic_ratio"]
    min_split = p["min_split"]
    X = svd_embed[:, :d]
    basename = (
        f"-method={omni_method}-d={d}-bic_ratio={bic_ratio}-min_split={min_split}"
    )
    title = f"Method={omni_method}, d={d}, BIC ratio={bic_ratio}-min_split={min_split}"

    currtime = time.time()

    np.random.seed(8888)
    mc = BinaryCluster(
        "0",
        adj=adj,  # stored for plotting, basically
        n_init=50,  # number of initializations for GMM at each stage
        meta=new_meta,  # stored for plotting and adding labels
        stashfig=stashfig,  # for saving figures along the way
        X=X,  # input data that actually matters
        bic_ratio=bic_ratio,
        reembed=False,
        min_split=min_split,
    )

    mc.fit(n_levels=n_levels, metric=metric)
    print(f"{(time.time() - currtime)/60:0.2f} minutes elapsed for clustering")

    inds = np.concatenate((lp_inds, rp_inds))
    cluster_adj = adj[np.ix_(inds, inds)]
    cluster_meta = mc.meta
    cluster_meta["sf"] = -signal_flow(cluster_adj)  # for some of the sorting

    # save results
    stashcsv(cluster_meta, "meta" + basename)
    adj_df = pd.DataFrame(
        cluster_adj, index=cluster_meta.index, columns=cluster_meta.index
    )
    stashcsv(adj_df, "adj" + basename)

    # plot results
    lowest_level = 7  # last level to show for dendrograms, adjacencies
    plot_clustering_results(
        cluster_adj, cluster_meta, basename, lowest_level=lowest_level
    )
    print()
