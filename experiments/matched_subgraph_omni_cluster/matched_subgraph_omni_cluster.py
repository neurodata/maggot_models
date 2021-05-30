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
from graspologic.embed import OmnibusEmbed, selectSVD
from graspologic.models import DCSBMEstimator, SBMEstimator
from graspologic.utils import (
    augment_diagonal,
    binarize,
    pass_to_ranks,
    remove_loops,
    to_laplacian,
)
from scipy.stats import poisson
from src.data import load_maggot_graph


from src.cluster import BinaryCluster
from src.data import load_palette
from src.data.get_data import join_node_meta
from src.graph import MetaGraph
from src.io import savecsv, savefig
from src.visualization import (
    CLASS_COLOR_DICT,
    add_connections,
    adjplot,
    plot_color_labels,
    plot_double_dendrogram,
    plot_single_dendrogram,
    set_theme,
)
from giskard.utils import get_paired_inds

# For saving outputs
FNAME = os.path.basename(__file__)[:-3]
print(FNAME)


set_theme()
palette = load_palette()

np.random.seed(8888)

save_path = Path("maggot_models/experiments/matched_subgraph_omni_cluster/")


def stashfig(name, **kws):
    savefig(name, pathname=save_path / "figs", format="png", dpi=300, **kws)


def stashcsv(df, name, **kws):
    savecsv(df, name, pathname=save_path / "outs", **kws)


def plot_pairs(
    X, labels, model=None, left_pair_inds=None, right_pair_inds=None, equal=False
):
    """Plots pairwise dimensional projections, and draws lines between known pair neurons

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
    data = pd.DataFrame(data=X, columns=[str(i) for i in range(n_dims)])
    data["label"] = labels

    for i in range(n_dims):
        for j in range(n_dims):
            ax = axs[i, j]
            ax.axis("off")
            if i < j:
                sns.scatterplot(
                    data=data,
                    x=str(j),
                    y=str(i),
                    ax=ax,
                    alpha=0.7,
                    linewidth=0,
                    s=8,
                    legend=False,
                    hue="label",
                    palette=palette,
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
    # mg = sort_mg(mg, level_names)
    # meta = mg.meta
    # adj = mg.adj

    # set up figure
    # analysis, bars, colors, graph graph graph...
    # n_col = 1 + 2 + 1 + lowest_level + 1
    # n_row = 6
    # width_ratios = 4 * [1] + (lowest_level + 1) * [1.5]

    # fig = plt.figure(
    #     constrained_layout=False, figsize=(5 * 4 + (lowest_level + 1) * 8.5, 20)
    # )
    # gs = plt.GridSpec(nrows=n_row, ncols=n_col, figure=fig, width_ratios=width_ratios)

    # # plot the dendrograms
    # dend_axs = []
    # dend_axs.append(fig.add_subplot(gs[:, 1]))  # left
    # dend_axs.append(fig.add_subplot(gs[:, 2]))  # right
    # dend_axs.append(fig.add_subplot(gs[:, 3]))  # colormap
    # plot_double_dendrogram(meta, dend_axs[:-1], lowest_level=lowest_level)
    # plot_color_labels(meta, dend_axs[-1])

    # # plot the adjacency matrices for data and sampled data
    # adj_axs = np.empty((2, lowest_level + 1), dtype="O")
    # offset = 4
    # for level in np.arange(lowest_level + 1):
    #     ax = fig.add_subplot(gs[: n_row // 2, level + offset])
    #     adj_axs[0, level] = ax
    #     ax = fig.add_subplot(gs[n_row // 2 :, level + offset])
    #     adj_axs[1, level] = ax
    # plot_adjacencies(mg, adj_axs, lowest_level=lowest_level)

    # # TODO this should only deal with TRUE PAIRS
    # temp_meta = mg.meta.copy()
    # # all valid TRUE pairs
    # temp_meta = temp_meta[temp_meta["pair_id"] != -1]
    # # throw out anywhere the TRUE pair on other hemisphere doesn't exist
    # temp_meta = temp_meta[temp_meta["pair"].isin(mg.meta.index)]
    # temp_meta = temp_meta.sort_values(["hemisphere", "pair_id"], ascending=True)
    # n_true_pairs = len(temp_meta) // 2
    # assert (
    #     temp_meta.iloc[:n_true_pairs]["pair_id"].values
    #     == temp_meta.iloc[n_true_pairs:]["pair_id"].values
    # ).all()
    # temp_mg = mg.copy()
    # temp_mg = temp_mg.reindex(temp_meta.index, use_ids=True)
    # # mg = mg.sort_values(["hemisphere", "pair_id"], ascending=True)
    # meta = temp_mg.meta
    # adj = temp_mg.adj
    # n_pairs = len(meta) // 2
    # lp_inds = np.arange(n_pairs)
    # rp_inds = np.arange(n_pairs) + n_pairs
    # n_levels = 9  # how many levels to show in the curve plots

    # # plot the pairedness in the top left
    # palette = sns.color_palette("deep", 2)
    # sns.set_palette(palette)
    # ax = fig.add_subplot(gs[:2, 0])
    # plot_pairedness(meta, lp_inds, rp_inds, ax, n_levels=n_levels)

    # # plot the likelihood curves in the middle left
    # palette = sns.color_palette("deep")
    # palette = [palette[2], palette[4]]  # green, purple,
    # sns.set_palette(palette)
    # ax = fig.add_subplot(gs[2:4, 0], sharex=ax)
    # plot_model_liks(adj, meta, lp_inds, rp_inds, ax, n_levels=n_levels)

    # # plot the number of clusters in the bottom left
    # palette = sns.color_palette("deep")
    # palette = [palette[5]]  # brown
    # sns.set_palette(palette)
    # ax = fig.add_subplot(gs[4:6, 0], sharex=ax)
    # plot_cluster_size(meta, ax, n_levels=n_levels)

    # # finish up
    # plt.tight_layout()
    # stashfig(f"megafig-lowest={lowest_level}" + basename)
    # plt.close()

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
# graph_type = "G"
# VERSION = "2020-06-10"

# pair_meta = pd.read_csv(
#     "maggot_models/experiments/graph_match/outs/pair_meta.csv", index_col=0
# )

# master_mg = load_metagraph(graph_type)
# master_mg = master_mg.reindex(pair_meta.index, use_ids=True)
# master_mg = MetaGraph(master_mg.adj, pair_meta)

master_mg = load_maggot_graph()

# mg = MetaGraph(master_mg.adj, master_mg.meta)
mg = master_mg.copy()
mg = mg[mg.nodes["paper_clustered_neurons"] | mg.nodes["accessory_neurons"]].copy()
mg.to_largest_connected_component()
meta = mg.nodes.copy()
meta["inds"] = range(len(meta))
adj = mg.sum.adj.copy()


#%%
# inds on left and right that correspond to the bilateral pairs
# here these are the union of true and predicted paris, though
pseudo_lp_inds, pseudo_rp_inds = get_paired_inds(
    meta, check_in=False, pair_key="predicted_pair", pair_id_key="predicted_pair_id"
)
left_inds = meta[meta["left"]]["inds"]

print(f"Neurons left after preprocessing: {len(mg)}")


# %% [markdown]
# ## Load the 4-color graphs
# convention is "ad" = axon -> dendrite and so on...
edge_types = ["ad", "aa", "dd", "da"]  # "Gs"]
adjs = []
for et in edge_types:
    temp_mg = mg.to_edge_type_graph(et)
    # # this line is important, to make the graphs aligned
    # temp_mg.reindex(mg.meta.index, use_ids=True)
    temp_adj = temp_mg.adj
    adjs.append(temp_adj)

# %% [markdown]
# ## SVDer
# 8, 16 seems to work
n_omni_components = 16  # this is used for all of the embedings initially
n_svd_components = 16  # this is for the last step
method = "ase"  # one could also do LSE


def svd(X, n_components=n_svd_components):
    return selectSVD(X, n_components=n_components, algorithm="full")[0]


# %% [markdown]
# # Run Omni and deal with latent positions appropriately

omni_method = "color_iso"
# use the predicted pairs for this part
lp_inds = pseudo_lp_inds
rp_inds = pseudo_rp_inds
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

# %% [markdown]
# ##

# since we used lp_inds and rp_inds to index the adjs, we need to reindex the meta
n_pairs = len(lp_inds)
new_lp_inds = np.arange(n_pairs)
new_rp_inds = np.arange(n_pairs) + n_pairs
new_meta = meta.iloc[np.concatenate((lp_inds, rp_inds), axis=0)].copy()
labels = new_meta["simple_group"].values


# columns = [
#     "hemisphere",
#     "simple_group",
#     "merge_class",
#     "pair",
#     "pair_id",
#     "lineage",
#     "name",
# ]
# embedding_out = "maggot_models/experiments/matched_subgraph_omni_cluster/outs/omni_"

# save_meta = new_meta[columns].copy()
# save_meta.index.name = "skid"
# save_meta = save_meta.reset_index()
# save_meta.set_index("name")

# tensor_projection_writer(
#     embedding_out + "embed",
#     embedding_out + "labels",
#     svd_embed,
#     save_meta.values.tolist(),
# )
# save_meta.to_csv(embedding_out + "labels", sep="\t")

# plot_pairs(
#     svd_embed[:, :8], labels, left_pair_inds=new_lp_inds, right_pair_inds=new_rp_inds
# )
# stashfig(f"pairs-method={omni_method}")


# simple_umap_scatterplot(
#     svd_embed, labels=labels, metric="cosine", palette=palette, title="Bilateral Omni"
# )
# stashfig(f"umap-method={omni_method}")


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
    # {"d": 8, "bic_ratio": 0, "min_split": 32},
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
    basename = f"divisive-omni-d={d}-bic_ratio={bic_ratio}-min_split={min_split}"
    title = f"Method={omni_method}, d={d}, BIC ratio={bic_ratio}-min_split={min_split}"

    currtime = time.time()

    np.random.seed(8888)
    mc = BinaryCluster(
        "0",
        adj=adj,  # stored for plotting, basically
        n_init=1,  # 50  # number of initializations for GMM at each stage
        meta=new_meta,  # stored for plotting and adding labels
        stashfig=stashfig,  # for saving figures along the way
        X=X,  # input data that actually matters
        bic_ratio=bic_ratio,
        reembed=False,
        min_split=min_split,
    )

    mc.fit(n_levels=n_levels, metric=metric)
    print(f"{(time.time() - currtime)/60:0.2f} minutes elapsed for clustering")

    cluster_meta = mc.meta
    cols = [col for col in cluster_meta.columns if "lvl" in col]
    print(cols)
    join_node_meta(cluster_meta[cols], overwrite=True)
    # inds = np.concatenate((lp_inds, rp_inds))
    # cluster_adj = adj[np.ix_(inds, inds)]
    # cluster_meta = mc.meta
    # cluster_meta["sf"] = -signal_flow(cluster_adj)  # for some of the sorting

    # # save results
    # stashcsv(cluster_meta, "meta" + basename)
    # adj_df = pd.DataFrame(
    #     cluster_adj, index=cluster_meta.index, columns=cluster_meta.index
    # )
    # stashcsv(adj_df, "adj" + basename)

    # # plot results
    # lowest_level = 7  # last level to show for dendrograms, adjacencies
    # plot_clustering_results(
    #     cluster_adj, cluster_meta, basename, lowest_level=lowest_level
    # )
    print()
