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
        pathname="./maggot_models/experiments/evaluate_blockmodel/figs",
        **kws,
    )


#%%
CLUSTER_KEYS = []
gt_keys = ["gt_blockmodel_labels"]
CLUSTER_KEYS += gt_keys
agglom_keys = [
    "agglom_labels_t=2_n_components=64",
    "agglom_labels_t=2.25_n_components=64",
    "agglom_labels_t=2.5_n_components=64",
    "agglom_labels_t=2.75_n_components=64",
    "agglom_labels_t=3_n_components=64",
]
CLUSTER_KEYS += agglom_keys
gaussian_keys = []
CLUSTER_KEYS += gaussian_keys

push = 2
red_shades = sns.color_palette("Reds", n_colors=len(gt_keys) + push)[push:][::-1]
blue_shades = sns.color_palette("Blues", n_colors=len(agglom_keys) + push)[push:][::-1]
# green_shades = sns.color_palette("Greens", n_colors=len(alphas) + push)[push:]

shades = red_shades + blue_shades
palette = dict(zip(CLUSTER_KEYS, shades))
palette

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

        n_params = estimator._n_parameters() + len(uni_labels)

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
            )
        )

        estimator = model(directed=True, loops=False)
        estimator.fit(right_adj, inv[rp_inds])
        train_right_p = estimator.p_mat_
        if pad_probabilities:
            train_right_p[train_right_p == 0] = 1 / train_right_p.size

        n_params = estimator._n_parameters() + len(uni_labels)

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
fig, ax = plt.subplots(1, 1, figsize=(8, 6))
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
ax.set(ylabel="Likelihood (hemisphere swapped)", yticks=[], xlabel="# parameters")
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
