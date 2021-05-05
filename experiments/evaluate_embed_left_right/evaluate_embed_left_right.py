#%%
import datetime
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from giskard.plot import crosstabplot
from giskard.utils import get_paired_inds
from graspologic.cluster import DivisiveCluster
from graspologic.cluster.autogmm import _labels_to_onehot, _onehot_to_initial_params
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score
from sklearn.mixture import GaussianMixture
from src.data import join_node_meta, load_maggot_graph
from src.visualization import CLASS_COLOR_DICT, set_theme
from src.io import savefig

set_theme()

t0 = time.time()
CLASS_KEY = "merge_class"
palette = CLASS_COLOR_DICT

mg = load_maggot_graph()
mg = mg[mg.nodes["has_embedding"]]
mg = mg[mg.nodes["pair_id"] > 1]
nodes = mg.nodes

out_path = Path("./maggot_models/experiments/evaluate_embed_left_right")

FORMAT = "png"


def stashfig(name, format=FORMAT, **kws):
    savefig(
        name, pathname=out_path / "figs", format=format, dpi=300, save_on=True, **kws
    )


lp_inds, rp_inds = get_paired_inds(nodes)

#%%


def initialize_gmm(labels, X, cov_type, **kwargs):
    onehot = _labels_to_onehot(labels)
    weights, means, precisions = _onehot_to_initial_params(X, onehot, cov_type)
    gm = GaussianMixture(
        n_components=len(weights),
        covariance_type=cov_type,
        weights_init=weights,
        means_init=means,
        precisions_init=precisions,
    )
    return gm


nodes["inds"] = range(len(nodes))
n_components = 12
latent_cols = [f"latent_{i}" for i in range(n_components)]
X = nodes[latent_cols].values

n_resamples = 10
resample_prop = 0.9
n_per_sample = int(np.ceil(resample_prop * len(nodes) / 2))
models = {}
datas = {}
for side in ["L", "R"]:
    side_nodes = nodes[nodes["hemisphere"] == side]
    for i in range(n_resamples):

        choices = np.random.choice(len(side_nodes), size=n_per_sample, replace=False)
        for n_clusters in np.arange(60, 70, 10):
            print((side, i, n_clusters))
            fit_inds = side_nodes.iloc[choices]["inds"].values
            agg = AgglomerativeClustering(
                n_clusters=n_clusters, affinity="cosine", linkage="average"
            )
            sub_X = X[fit_inds]
            pred_labels = agg.fit_predict(sub_X)
            gmm = initialize_gmm(pred_labels, sub_X, cov_type="full")
            gmm.fit(sub_X)
            models[(side, i, n_clusters)] = gmm
            datas[(side, i, n_clusters)] = sub_X

results = []
for train_params, model in models.items():
    for test_params, data in datas.items():
        if train_params[2] == test_params[2]:
            score = model.bic(X)
            result = {}
            result["train_side"] = train_params[0]
            result["train_i"] = train_params[1]
            result["n_clusters"] = train_params[2]

            result["test_side"] = test_params[0]
            result["test_i"] = test_params[1]

            result["score"] = score

            results.append(result)
results = pd.DataFrame(results)
results

# #%%
# from graspologic.plot import pairplot_with_gmm

# pairplot_with_gmm(X, gmm)

#%%

sub_results = results[results["n_clusters"] == 60].drop("n_clusters", axis=1)
square_results = pd.pivot_table(
    sub_results, index=["train_side", "train_i"], columns=["test_side", "test_i"]
)
square_results

#%%

fig, ax = plt.subplots(1, 1, figsize=(8, 8))
sns.heatmap(square_results, square=True, cbar_kws=dict(shrink=0.6), ax=ax)
stashfig("pairwise-bic")
# rows.append(

#     {"i": i, "n_clusters": n_clusters, "side": "left", "pred_labels": left_pred}
# )
# rows.append(
#     {
#         "i": i,
#         "n_clusters": n_clusters,
#         "side": "right",
#         "pred_labels": right_pred,
#     }
# )

# for i, row1 in enumerate(rows):
#     for j, row2 in enumerate(rows):
#         if i < j:
#             adjusted_rand_score(row1["pred_labels"], row2["pred_labels"])
