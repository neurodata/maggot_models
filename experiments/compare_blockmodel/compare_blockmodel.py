#%%
import datetime
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from giskard.utils import get_paired_inds
from graspologic.models import DCSBMEstimator, SBMEstimator
from graspologic.utils import binarize, remove_loops
from scipy.stats import poisson
from src.data import join_node_meta, load_maggot_graph
from src.io import savefig
from src.visualization import set_theme

t0 = time.time()
set_theme()


def stashfig(name, **kws):
    savefig(
        name,
        pathname="./maggot_models/experiments/compare_blockmodel/figs",
        **kws,
    )


#%%
CLUSTER_KEYS = []
gt_keys = ["gt_blockmodel_labels"]
CLUSTER_KEYS += gt_keys
# agglom_keys = [
#     "agglom_labels_t=2_n_components=64",
#     "agglom_labels_t=2.25_n_components=64",
#     "agglom_labels_t=2.5_n_components=64",
#     "agglom_labels_t=2.75_n_components=64",
#     "agglom_labels_t=3_n_components=64",
# ]
# agglom_keys = [
#     "agglom_labels_t=0.6_n_components=64",
#     "agglom_labels_t=0.625_n_components=64",
#     "agglom_labels_t=0.65_n_components=64",
#     "agglom_labels_t=0.7_n_components=64",
#     "agglom_labels_t=0.75_n_components=64",
# ]
agglom_keys = [
    "agglom_labels_t=0.45_n_components=32",
    "agglom_labels_t=0.5_n_components=32",
    "agglom_labels_t=0.6_n_components=32",
    "agglom_labels_t=0.7_n_components=32",
]
CLUSTER_KEYS += agglom_keys
agglom_new_keys = [
    "cluster_agglom_K=50",
    "cluster_agglom_K=60",
    "cluster_agglom_K=70",
    "cluster_agglom_K=80",
    "cluster_agglom_K=90",
    "cluster_agglom_K=100",
    "cluster_agglom_K=110",
]
CLUSTER_KEYS += agglom_new_keys[::-1]
gaussian_keys = [
    "dc_level_4_n_components=10_min_split=32",
    "dc_level_5_n_components=10_min_split=32",
    "dc_level_6_n_components=10_min_split=32",
    "dc_level_7_n_components=10_min_split=32",
]
CLUSTER_KEYS += gaussian_keys
consensus_keys = [
    "co_cluster_n_clusters=85",
]
CLUSTER_KEYS += consensus_keys


push = 2
red_shades = sns.color_palette("Reds", n_colors=len(gt_keys) + push)[push:][::-1]
blue_shades = sns.color_palette("Blues", n_colors=len(agglom_keys) + push)[push:][::-1]
purple_shades = sns.color_palette("Purples", n_colors=len(agglom_new_keys) + push)[
    push:
][::-1]
green_shades = sns.color_palette("Greens", n_colors=len(gaussian_keys) + push)[push:][
    ::-1
]
grey_shades = sns.color_palette("Greys", n_colors=len(consensus_keys) + push)[push:][
    ::-1
]
shades = red_shades + blue_shades + purple_shades + green_shades + grey_shades
palette = dict(zip(CLUSTER_KEYS, shades))

#%%


def calculate_blockmodel_likelihood(
    adj, labels, lp_inds, rp_inds, pad_probabilities=True
):
    rows = []
    left_adj = binarize(adj[np.ix_(lp_inds, lp_inds)])
    left_adj = remove_loops(left_adj)
    right_adj = binarize(adj[np.ix_(rp_inds, rp_inds)])
    right_adj = remove_loops(right_adj)
    for model, name in zip([DCSBMEstimator, SBMEstimator], ["DCSBM", "SBM"]):
        estimator = model(directed=True, loops=False)
        uni_labels, inv = np.unique(labels, return_inverse=True)
        estimator.fit(left_adj, inv[lp_inds])
        train_left_p = estimator.p_mat_
        if pad_probabilities:
            train_left_p[train_left_p == 0] = 1 / train_left_p.size

        n_params = estimator._n_parameters() + len(labels)

        score = poisson.logpmf(left_adj, train_left_p).sum()
        rows.append(
            dict(
                train_side="Left",
                test="Same",
                test_side="Left",
                score=score,
                model=name,
                n_params=n_params,
                norm_score=score / left_adj.sum(),
                n_communities=len(uni_labels),
            )
        )
        score = poisson.logpmf(right_adj, train_left_p).sum()
        rows.append(
            dict(
                train_side="Left",
                test="Opposite",
                test_side="Right",
                score=score,
                model=name,
                n_params=n_params,
                norm_score=score / right_adj.sum(),
                n_communities=len(uni_labels),
            )
        )

        estimator = model(directed=True, loops=False)
        estimator.fit(right_adj, inv[rp_inds])
        train_right_p = estimator.p_mat_
        if pad_probabilities:
            train_right_p[train_right_p == 0] = 1 / train_right_p.size

        n_params = estimator._n_parameters() + len(labels)

        score = poisson.logpmf(left_adj, train_right_p).sum()
        rows.append(
            dict(
                train_side="Right",
                test="Opposite",
                test_side="Left",
                score=score,
                model=name,
                n_params=n_params,
                norm_score=score / left_adj.sum(),
                n_communities=len(uni_labels),
            )
        )
        score = poisson.logpmf(right_adj, train_right_p).sum()
        rows.append(
            dict(
                train_side="Right",
                test="Same",
                test_side="Right",
                score=score,
                model=name,
                n_params=n_params,
                norm_score=score / right_adj.sum(),
                n_communities=len(uni_labels),
            )
        )
    return pd.DataFrame(rows)


mg = load_maggot_graph()
mg = mg[mg.nodes["has_embedding"]]
nodes = mg.nodes
adj = mg.sum.adj
lp_inds, rp_inds = get_paired_inds(nodes)
rows = []
for cluster_key in CLUSTER_KEYS:
    labels = nodes[cluster_key]
    cluster_results = calculate_blockmodel_likelihood(adj, labels, lp_inds, rp_inds)
    cluster_results["cluster_method"] = cluster_key
    rows.append(cluster_results)
results = pd.concat(rows, ignore_index=True)

# %%


def n_params2k(n_params):
    k = n_params - len(adj)
    k = np.sqrt(k)
    return k


def k2n_params(k):
    n_params = k ** 2 + 2 * len(adj)
    return n_params


fig, axs = plt.subplots(1, 2, figsize=(12, 6))

ymin = results[results["model"] == "DCSBM"]["norm_score"].min()
ymax = results[results["model"] == "DCSBM"]["norm_score"].max()
yrange = ymax - ymin
ymax += yrange * 0.02
ymin -= yrange * 0.02
ylim = (ymin, ymax)

ax = axs[0]
select_results = results[results["model"] == "DCSBM"].copy()
select_results = select_results.groupby(["cluster_method", "test"]).mean().reset_index()
select_results = select_results[select_results["test"] == "Same"]
sns.scatterplot(
    data=select_results,
    x="n_params",
    y="norm_score",
    hue="cluster_method",
    hue_order=CLUSTER_KEYS,
    ax=ax,
    palette=palette,
)
ax.get_legend().remove()
ax.set(
    ylabel="Likelihood (same hemisphere)", yticks=[], xlabel="# parameters", ylim=ylim
)

sec_ax = ax.secondary_xaxis(-0.2, functions=(n_params2k, k2n_params))
sec_ax.set_xlabel("# of communities")

ax = axs[1]
select_results = results[results["model"] == "DCSBM"].copy()
select_results = select_results.groupby(["cluster_method", "test"]).mean().reset_index()
select_results = select_results[select_results["test"] == "Opposite"]
sns.scatterplot(
    data=select_results,
    x="n_params",
    y="norm_score",
    hue="cluster_method",
    hue_order=CLUSTER_KEYS,
    ax=ax,
    palette=palette,
)
ax.get_legend().remove()
ax.set(
    ylabel="Likelihood (hemisphere swapped)",
    yticks=[],
    xlabel="# parameters",
    ylim=ylim,
)


sec_ax = ax.secondary_xaxis(-0.2, functions=(n_params2k, k2n_params))
sec_ax.set_xlabel("# of communities")
stashfig("lik-by-n_params-blind")
ax.legend(bbox_to_anchor=(1, 1), loc="upper left")
stashfig("lik-by-n_params")


#%%
elapsed = time.time() - t0
delta = datetime.timedelta(seconds=elapsed)
print("----")
print(f"Script took {delta}")
print(f"Completed at {datetime.datetime.now()}")
print("----")
