# %% [markdown]
# ##
import os
import time
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import poisson

from graspy.embed import OmnibusEmbed, selectSVD
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
    set_theme,
)
from topologic.io import tensor_projection_writer

# For saving outputs
FNAME = os.path.basename(__file__)[:-3]
print(FNAME)


set_theme()

np.random.seed(8888)

save_path = Path("maggot_models/experiments/evaluate_clustering/")

CLASS_KEY = "merge_class"
CLASS_ORDER = "median_node_visits"


def stashfig(name, fmt="pdf", **kws):
    savefig(name, pathname=save_path / "figs", fmt=fmt, save_on=True, **kws)


def stashcsv(df, name, **kws):
    savecsv(df, name, pathname=save_path / "outs", **kws)


def sort_mg(mg, level_names, class_order=CLASS_ORDER):
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
    class_order = [class_order]
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
            sort_class=level_names[: level + 1],
            item_order=[f"{CLASS_KEY}_{CLASS_ORDER}_order", CLASS_KEY, CLASS_ORDER],
            class_order=CLASS_ORDER,
            meta=full_mg.meta,
            palette=CLASS_COLOR_DICT,
            colors=CLASS_KEY,
            ticks=False,
            gridline_kws=dict(linewidth=0, color="grey", linestyle="--"),  # 0.2
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
            sort_class=level_names[: level + 1],
            item_order=[f"{CLASS_KEY}_{CLASS_ORDER}_order", CLASS_KEY, CLASS_ORDER],
            class_order=CLASS_ORDER,
            meta=full_mg.meta,
            palette=CLASS_COLOR_DICT,
            colors=CLASS_KEY,
            ticks=False,
            gridline_kws=dict(linewidth=0, color="grey", linestyle="--"),  # 0.2
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
    # handles, labels = ax.get_legend_handles_labels()
    # labels[0] = "Test side"
    # labels[3] = "Fit side"
    # ax.legend(handles=handles, labels=labels, bbox_to_anchor=(0, 1), loc="upper left")
    ax.set_ylabel(f"{model_name} normalized log lik.")
    ax.set_yticks([])
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
            text=max_size,
            xy=(l, max_size),
            xytext=(0, 30),
            ha="center",
            va="bottom",
            textcoords="offset pixels",
            fontsize=0.75 * fontsize,
        )
        ax.annotate(
            text=min_size,
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


def plot_clustering_results(
    adj,
    meta,
    basename,
    lowest_level=7,
    show_adjs=True,
    show_singles=True,
    make_flippable=False,
):
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
    dend_axs.append(fig.add_subplot(gs[2:4, 3]))  # colormap
    plot_double_dendrogram(
        meta,
        dend_axs[:-1],
        lowest_level=lowest_level,
        color_order=CLASS_ORDER,
        make_flippable=make_flippable,
    )
    plot_color_labels(meta, dend_axs[-1], color_order=CLASS_ORDER)

    # plot the adjacency matrices for data and sampled data
    if show_adjs:
        adj_axs = np.empty((2, lowest_level + 1), dtype="O")
        offset = 4
        for level in np.arange(lowest_level + 1):
            ax = fig.add_subplot(gs[: n_row // 2, level + offset])
            adj_axs[0, level] = ax
            ax = fig.add_subplot(gs[n_row // 2 :, level + offset])
            adj_axs[1, level] = ax

        plot_adjacencies(mg, adj_axs, lowest_level=lowest_level)

    temp_meta = mg.meta.copy()
    # all valid TRUE pairs
    temp_meta = temp_meta[temp_meta["pair_id"] != -1]
    # throw out anywhere the TRUE pair on other hemisphere doesn't exist
    temp_meta = temp_meta[temp_meta["pair"].isin(mg.meta.index)]
    temp_meta = temp_meta.sort_values(["hemisphere", "pair_id"], ascending=True)
    n_true_pairs = len(temp_meta) // 2
    assert (
        temp_meta.iloc[:n_true_pairs]["pair_id"].values
        == temp_meta.iloc[n_true_pairs:]["pair_id"].values
    ).all()
    temp_mg = mg.copy()
    temp_mg = temp_mg.reindex(temp_meta.index, use_ids=True)
    # mg = mg.sort_values(["hemisphere", "pair_id"], ascending=True)
    meta = temp_mg.meta
    adj = temp_mg.adj
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
    stashfig(f"megafig-lowest={lowest_level}" + basename, fmt="png")
    plt.close()

    if show_singles:
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
# ##
omni_method = "color_iso"
d = 8
bic_ratio = 0.95
min_split = 32

basename = f"-method={omni_method}-d={d}-bic_ratio={bic_ratio}-min_split={min_split}"
meta = pd.read_csv(
    f"maggot_models/experiments/matched_subgraph_omni_cluster/outs/meta{basename}.csv",
    index_col=0,
)
meta["lvl0_labels"] = meta["lvl0_labels"].astype(str)
adj_df = pd.read_csv(
    f"maggot_models/experiments/matched_subgraph_omni_cluster/outs/adj{basename}.csv",
    index_col=0,
)
adj = adj_df.values

name_map = {
    "Sens": "Sensory",
    "LN": "Local",
    "PN": "Projection",
    "KC": "Kenyon cell",
    "LHN": "Lateral horn",
    "MBIN": "MBIN",
    "Sens2o": "2nd order sensory",
    "unk": "Unknown",
    "MBON": "MBON",
    "FBN": "MB feedback",
    "CN": "Convergence",
    "PreO": "Pre-output",
    "Outs": "Output",
    "Motr": "Motor",
}
meta["simple_class"] = meta["simple_class"].map(name_map)
print(meta["simple_class"].unique())
meta["merge_class"] = meta["simple_class"]  # HACK


graph_type = "Gad"
n_init = 256
max_hops = 16
allow_loops = False
walk_spec = f"gt={graph_type}-n_init={n_init}-hops={max_hops}-loops={allow_loops}"
walk_meta = pd.read_csv(
    f"maggot_models/experiments/walk_sort/outs/meta_w_order-{walk_spec}.csv",
    index_col=0,
)
meta["median_node_visits"] = walk_meta["median_node_visits"]

# %%
# plot results
lowest_level = 7  # last level to show for dendrograms, adjacencies
plot_clustering_results(
    adj,
    meta,
    basename,
    lowest_level=lowest_level,
    show_adjs=True,
    show_singles=False,
    make_flippable=False,
)

#%%
# lowest_level = 7
# mg = MetaGraph(adj, meta)
# level_names = [f"lvl{i}_labels" for i in range(lowest_level + 1)]
# mg = sort_mg(mg, level_names)
# fig, axs = plt.subplots(
#     2, lowest_level + 1, figsize=10 * np.array([lowest_level + 1, 2])
# )
# for level in np.arange(lowest_level + 1):
#     plot_adjacencies(mg, axs, lowest_level=lowest_level)
# stashfig(f"adjplots-lowest={lowest_level}" + basename, fmt="png")

# %% [markdown]
# # ##

# mg = MetaGraph(adj, meta)
# level_names = [f"lvl{i}_labels" for i in range(lowest_level + 1)]
# meta = mg.meta
# sort_class = level_names + ["merge_class"]
# class_order = ["sf"]
# total_sort_by = []
# for sc in sort_class:
#     for co in class_order:
#         class_value = meta.groupby(sc)[co].mean()
#         meta[f"{sc}_{co}_order"] = meta[sc].map(class_value)
#         total_sort_by.append(f"{sc}_{co}_order")
#     total_sort_by.append(sc)
# mg = mg.sort_values(total_sort_by, ascending=False)
# mg.meta[["merge_class", "simple_class", "lvl0_labels_sf_order"]]

# # %% [markdown]
# # ##

# sorted_meta = meta.groupby(total_sort_by)["sf"].mean()
# # sorted_meta.head(20)
# meta.groupby(["lvl0_labels", "lvl1_labels", "merge_class"], sort=False)[
#     "merge_class_sf_order"
# ].mean()
# # sizes = meta.groupby([leaf_key, "merge_class"], sort=False).size()

# # %%
# meta.groupby(["lvl0_labels", "merge_class"], sort=False)["merge_class_sf_order"].mean()


# # %%
