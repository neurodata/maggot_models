#%%
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import src.utils as utils

import numpy as np

#%% [markdown]
# ### Choose experiment, print out configurations

#%%# Load SBM
base_path = "./maggot_models/models/runs/"
experiment = "fit_sbm"
run = 1
config = utils.load_config(base_path, experiment, run)
sbm_left_df = utils.load_pickle(base_path, experiment, run, "sbm_left_df")
sbm_right_df = utils.load_pickle(base_path, experiment, run, "sbm_right_df")

#%% Load coSBM
base_path = "./maggot_models/models/runs/"
experiment = "fit_cosbm"
run = 1
config = utils.load_config(base_path, experiment, run)
cosbm_left_df = utils.load_pickle(base_path, experiment, run, "cosbm_left_df")
cosbm_right_df = utils.load_pickle(base_path, experiment, run, "cosbm_right_df")


#%%
def get_best(df, param_name="param_n_components", score_name="mse"):
    param_range = np.unique(df[param_name].values)
    best_rows = []
    for param_value in param_range:
        temp_df = df[df[param_name] == param_value]
        ind = temp_df[score_name].idxmin()  # this is the metric we are choosing on
        best_rows.append(temp_df.loc[ind, :])
    return pd.DataFrame(best_rows)


best_sbm_left_df = get_best(sbm_left_df, param_name="param_n_blocks", score_name="mse")
best_sbm_right_df = get_best(
    sbm_right_df, param_name="param_n_blocks", score_name="mse"
)
best_cosbm_left_df = get_best(
    cosbm_left_df, param_name="param_n_blocks", score_name="mse"
)
best_cosbm_right_df = get_best(
    cosbm_right_df, param_name="param_n_blocks", score_name="mse"
)

#%%
# Plotting setup}
plt.style.use("seaborn-white")
sns.set_context("talk", font_scale=1.5)
sns.set_palette("Set1")
plt_kws = dict(s=75, linewidth=0, legend="brief")
figsize = (20, 10)
cmap = sns.light_palette("purple", as_cmap=True)
save_dir = Path("./maggot_models/reports/figures/vertex_agnostic_mse")
#%% Figures - rank SBMs


plt.figure(figsize=figsize)
sns.scatterplot(data=best_sbm_left_df, x="n_params", y="mse", label="SBM", **plt_kws)
sns.scatterplot(
    data=best_cosbm_left_df, x="n_params", y="mse", label="COSBM", **plt_kws
)
plt.xlabel("# Params (SBM params for SBMs)")
plt.ylabel("MSE")
plt.title(f"Left")
plt.legend()
plt.show()
# plt.savefig(save_dir / ".pdf", format="pdf", facecolor="w")

#%%
plt.figure(figsize=figsize)
sns.scatterplot(
    data=best_sbm_df,
    x="n_params_sbm",
    y="mse",
    hue="n_block_try",
    size="rank_try",
    palette=cmap,
    **plt_kws,
)
plt.xlabel("# Params (SBM params for SBMs)")
plt.ylabel("MSE")
plt.title(f"Drosophila old MB left, directed ({experiment}:{run})")
plt.savefig(save_dir / "rank_sbm_2.pdf", format="pdf", facecolor="w")


plt.figure(figsize=figsize)
sns.scatterplot(
    data=best_sbm_df,
    x="n_params_sbm",
    y="mse",
    hue="n_block_try",
    size="rank_proportion",
    palette=cmap,
    **plt_kws,
)
plt.xlabel("# Params (SBM params for SBMs)")
plt.ylabel("MSE")
plt.title(f"Drosophila old MB left, directed ({experiment}:{run})")
plt.savefig(save_dir / "rank_sbm_3.pdf", format="pdf", facecolor="w")

#%% Try to plot with lines instead
plt_kws = dict(legend="brief")
cmap = "jet"
plt.figure(figsize=figsize)
sns.lineplot(
    data=best_sbm_df, x="n_params_sbm", y="mse", hue="rank_try", palette=cmap, **plt_kws
)

plt.xlabel("# Params (SBM params for SBMs)")
plt.ylabel("MSE")
plt.title(f"Drosophila old MB left, directed ({experiment}:{run})")
plt.savefig(save_dir / "rank_sbm_ranklines.pdf", format="pdf", facecolor="w")

plt_kws = dict(legend="brief")
cmap = "jet"
plt.figure(figsize=figsize)
sns.lineplot(
    data=best_sbm_df,
    x="n_params_sbm",
    y="mse",
    hue="n_block_try",
    palette=cmap,
    **plt_kws,
)

plt.xlabel("# Params (SBM params for SBMs)")
plt.ylabel("MSE")
plt.title(f"Drosophila old MB left, directed ({experiment}:{run})")
plt.savefig(save_dir / "rank_sbm_Klines.pdf", format="pdf", facecolor="w")

# #%%
# from graspy.models import SBMEstimator
# from graspy.datasets import load_drosophila_left, load_drosophila_right
# from graspy.utils import binarize

# sbm = SBMEstimator(directed=True, loops=False)
# left_adj, left_labels = load_drosophila_left(return_labels=True)
# left_adj = binarize(left_adj)
# sbm.fit(left_adj, y=left_labels)
# sbm.mse(left_adj)
# sbm._n_parameters()

# right_adj, right_labels = load_drosophila_right(return_labels=True)

# er = SBMEstimator(directed=True, loops=False, n_blocks=2)
# er.fit(left_adj)
# er.mse(left_adj)
# heatmap(
#     left_adj, inner_hier_labels=er.vertex_assignments_, outer_hier_labels=left_labels
# )
