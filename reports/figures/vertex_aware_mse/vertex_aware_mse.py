#%%
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from IPython import get_ipython  # just to decieve flake8

import src.utils as utils


#%% [markdown]
# ### Choose experiment, print out configurations

#%% Load DCSBM
base_path = "./maggot_models/models/runs/"
experiment = "fit_dcsbm"
run = 8
config = utils.load_config(base_path, experiment, run)
dcsbm_left_df = utils.load_pickle(base_path, experiment, run, "dcsbm_left_df")
dcsbm_right_df = utils.load_pickle(base_path, experiment, run, "dcsbm_right_df")

#%% Load dDCSBM
base_path = "./maggot_models/models/runs/"
experiment = "fit_ddcsbm"
run = 3
config = utils.load_config(base_path, experiment, run)
ddcsbm_left_df = utils.load_pickle(base_path, experiment, run, "ddcsbm_left_df")
ddcsbm_right_df = utils.load_pickle(base_path, experiment, run, "ddcsbm_right_df")

#%% Load RDPG
base_path = "./maggot_models/models/runs/"
experiment = "fit_rdpg"
run = 3
config = utils.load_config(base_path, experiment, run)
rdpg_left_df = utils.load_pickle(base_path, experiment, run, "rdpg_left_df")
rdpg_right_df = utils.load_pickle(base_path, experiment, run, "rdpg_right_df")

#%%


def get_best(df, param_name="param_n_components", score_name="mse"):
    param_range = np.unique(df[param_name].values)
    best_rows = []
    for param_value in param_range:
        temp_df = df[df[param_name] == param_value]
        ind = temp_df[score_name].idxmin()  # this is the metric we are choosing on
        best_rows.append(temp_df.loc[ind, :])
    return pd.DataFrame(best_rows)


best_dcsbm_left_df = get_best(
    dcsbm_left_df, param_name="param_n_blocks", score_name="mse"
)
best_dcsbm_right_df = get_best(
    dcsbm_right_df, param_name="param_n_blocks", score_name="mse"
)
best_ddcsbm_left_df = get_best(
    ddcsbm_left_df, param_name="param_n_blocks", score_name="mse"
)
best_ddcsbm_right_df = get_best(
    ddcsbm_right_df, param_name="param_n_blocks", score_name="mse"
)
best_rdpg_left_df = get_best(
    rdpg_left_df, param_name="param_n_components", score_name="mse"
)
best_rdpg_right_df = get_best(
    rdpg_right_df, param_name="param_n_components", score_name="mse"
)

#%% Settings for all plots
plt.style.use("seaborn-white")
sns.set_context("talk", font_scale=1.5)
sns.set_palette("Set1")

title_fontsize = 40
axis_label_fontsize = 30
#%% Figure - show RDPG, DCSBM bests
fig, ax = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(20, 10))

plt_kws = dict(linewidth=0, y="mse", x="n_params", ax=ax[0])
sns.scatterplot(data=best_dcsbm_left_df, label="DCSBM", **plt_kws)
sns.scatterplot(data=best_ddcsbm_left_df, label="dDCSBM", **plt_kws)
sns.scatterplot(data=best_rdpg_left_df, label="RDPG", **plt_kws)
ax[0].set_title("Left", fontsize=title_fontsize)
ax[0].set_xscale("log")
ax[0].set_xlabel("# of parameters", fontsize=axis_label_fontsize)
ax[0].set_ylabel("MSE", fontsize=axis_label_fontsize)
plt.legend()

plt_kws = dict(linewidth=0, y="mse", x="n_params", ax=ax[1])
sns.scatterplot(data=best_dcsbm_right_df, label="DCSBM", **plt_kws)
sns.scatterplot(data=best_ddcsbm_right_df, label="dDCSBM", **plt_kws)
sns.scatterplot(data=best_rdpg_right_df, label="RDPG", **plt_kws)
ax[1].set_title("Right", fontsize=title_fontsize)
ax[1].set_xscale("log")
ax[1].set_xlabel("# of parameters", fontsize=axis_label_fontsize)
plt.tight_layout()
plt.legend()

plt.savefig(
    "./maggot_models/reports/figures/vertex_aware_mse/vertex_aware_mse.pdf",
    format="pdf",
    facecolor="w",
    bbox_inches="tight",
)
#%%
# plt.figure(figsize=(20, 10))
# plt_kws = dict(linewidth=0, y="likelihood", x="n_params")
# sns.scatterplot(data=best_dcsbm_df, label="DCSBM", **plt_kws)
# sns.scatterplot(data=best_ddcsbm_df, label="dDCSBM", **plt_kws)
# sns.scatterplot(data=best_rdpg_df, label="RDPG", **plt_kws)
# plt.legend()

# #%% Figure - observe the effect of regularization on RDPG fitting
# plot_df = rdpg_df[rdpg_df["param_plus_c_weight"] == 0]
# plt.figure(figsize=(20, 10))
# sns.scatterplot(
#     data=plot_df,
#     x="n_params",
#     y="mse",
#     hue="param_diag_aug_weight",
#     size="param_diag_aug_weight",
#     alpha=0.5,
#     linewidth=0,
# )

