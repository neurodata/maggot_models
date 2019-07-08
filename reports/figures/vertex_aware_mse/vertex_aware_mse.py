#%%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import src.utils as utils

#%% Load DCSBM
base_path = "./maggot_models/models/runs/"
experiment = "fit_dcsbm"
run = 8
config = utils.load_config(base_path, experiment, run)
dcsbm_left_df = utils.load_pickle(base_path, experiment, run, "dcsbm_left_df")
dcsbm_right_df = utils.load_pickle(base_path, experiment, run, "dcsbm_right_df")

#%% Load dDCSBM
experiment = "fit_ddcsbm"
run = 3
config = utils.load_config(base_path, experiment, run)
ddcsbm_left_df = utils.load_pickle(base_path, experiment, run, "ddcsbm_left_df")
ddcsbm_right_df = utils.load_pickle(base_path, experiment, run, "ddcsbm_right_df")

#%% Load RDPG
experiment = "fit_rdpg"
run = 3
config = utils.load_config(base_path, experiment, run)
rdpg_left_df = utils.load_pickle(base_path, experiment, run, "rdpg_left_df")
rdpg_right_df = utils.load_pickle(base_path, experiment, run, "rdpg_right_df")

#%% Load a priori
experiment = "fit_a_priori"
run = 1
config = utils.load_config(base_path, experiment, run)
ap_dcsbm_left_df = utils.load_pickle(base_path, experiment, run, "dcsbm_left_df")
ap_dcsbm_right_df = utils.load_pickle(base_path, experiment, run, "dcsbm_right_df")

ap_ddcsbm_left_df = utils.load_pickle(base_path, experiment, run, "ddcsbm_left_df")
ap_ddcsbm_right_df = utils.load_pickle(base_path, experiment, run, "ddcsbm_right_df")

#%%


def get_best(df, param_name="param_n_components", score_name="mse"):
    param_range = np.unique(df[param_name].values)
    best_rows = []
    for param_value in param_range:
        temp_df = df[df[param_name] == param_value]
        ind = temp_df[score_name].idxmin()  # this is the metric we are choosing on
        best_rows.append(temp_df.loc[ind, :])
    return pd.DataFrame(best_rows)


score_name = "mse"
best_dcsbm_left_df = get_best(
    dcsbm_left_df, param_name="param_n_blocks", score_name=score_name
)
best_dcsbm_right_df = get_best(
    dcsbm_right_df, param_name="param_n_blocks", score_name=score_name
)
best_ddcsbm_left_df = get_best(
    ddcsbm_left_df, param_name="param_n_blocks", score_name=score_name
)
best_ddcsbm_right_df = get_best(
    ddcsbm_right_df, param_name="param_n_blocks", score_name=score_name
)
best_rdpg_left_df = get_best(
    rdpg_left_df, param_name="param_n_components", score_name=score_name
)
best_rdpg_right_df = get_best(
    rdpg_right_df, param_name="param_n_components", score_name=score_name
)

#%% Settings for all plots
plt.style.use("seaborn-white")
sns.set_context("talk", font_scale=1.5)
sns.set_palette("Set1")

#%%
# build a colormap
base_cmap = sns.color_palette("Set1", n_colors=3, desat=1)
sns.palplot(base_cmap)
ap_cmap = sns.color_palette("Set1", desat=0.3, n_colors=2)
sns.palplot(ap_cmap)
custom_cmap = sns.color_palette(["#de2d26", "#3182bd", "#31a354", "#fb6a4a", "#6baed6"])
# custom_cmap = base_cmap + ap_cmap
sns.set_palette(custom_cmap)
#%%
title_fontsize = 40
axis_label_fontsize = 30
ap_marker = "X"
ap_size = 200
#%% Figure - show RDPG, DCSBM bests
# Left
fig, ax = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(20, 10))
plt_kws = dict(linewidth=0, y=score_name, x="n_params", ax=ax[0])

# model curves
sns.scatterplot(data=best_dcsbm_left_df, label="DCSBM", **plt_kws)
sns.scatterplot(data=best_ddcsbm_left_df, label="dDCSBM", **plt_kws)
sns.scatterplot(data=best_rdpg_left_df, label="RDPG", **plt_kws)

# a priori
sns.scatterplot(
    data=ap_dcsbm_left_df, label="priori DCSBM", marker=ap_marker, s=ap_size, **plt_kws
)
sns.scatterplot(
    data=ap_ddcsbm_left_df,
    label="priori dDCSBM",
    marker=ap_marker,
    s=ap_size,
    **plt_kws
)

ax[0].set_title("Left", fontsize=title_fontsize)
ax[0].set_xscale("log")
ax[0].set_xlabel("# of parameters", fontsize=axis_label_fontsize)
ax[0].set_ylabel("MSE", fontsize=axis_label_fontsize)
plt.legend()

# Right
plt_kws = dict(linewidth=0, y=score_name, x="n_params", ax=ax[1])

# model curves
sns.scatterplot(data=best_dcsbm_right_df, label="DCSBM", **plt_kws)
sns.scatterplot(data=best_ddcsbm_right_df, label="dDCSBM", **plt_kws)
sns.scatterplot(data=best_rdpg_right_df, label="RDPG", **plt_kws)

# a priori
sns.scatterplot(
    data=ap_dcsbm_right_df, label="priori DCSBM", marker=ap_marker, s=ap_size, **plt_kws
)
sns.scatterplot(
    data=ap_ddcsbm_right_df,
    label="priori dDCSBM",
    marker=ap_marker,
    s=ap_size,
    **plt_kws
)

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
