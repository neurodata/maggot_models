#%%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import src.utils as utils
from matplotlib import rc

# rc("text", usetex=True)

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

# build a colormap
base_cmap = sns.color_palette("Set1", n_colors=3, desat=1)
sns.palplot(base_cmap)
ap_cmap = sns.color_palette("Set1", desat=0.3, n_colors=2)
sns.palplot(ap_cmap)
cmap = sns.color_palette(["#de2d26", "#3182bd", "#31a354", "#fb6a4a", "#6baed6"])
sns.set_palette(cmap)

# Figure - show RDPG, DCSBM bests
title_fontsize = 40
axis_label_fontsize = 30
marker_label_fontsize = 25
ap_marker = "x"
ap_size = 250
fig, ax = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(20, 10))


def plot_arrow(ax, point, label, color, offset=(50, 50), marker_label_fontsize=30):
    arrowprops = {
        "arrowstyle": "-|>",
        "mutation_scale": 15,
        "relpos": (0, 0),
        "color": color,
        "shrinkB": 15,
        "shrinkA": 0,
        "linewidth": 5,
    }
    annotate_kws = dict(
        xy=point,
        xytext=offset,
        textcoords="offset points",
        xycoords="data",
        fontsize=marker_label_fontsize,
        c=color,
        arrowprops=arrowprops,
    )
    ax.annotate(label, **annotate_kws)


# Left
#
#

# model curves
plt_kws = dict(linewidth=0, y=score_name, x="n_params", ax=ax[0], legend=False)
sns.scatterplot(data=best_dcsbm_left_df, label="DCSBM", **plt_kws)
sns.scatterplot(data=best_ddcsbm_left_df, label="dDCSBM", **plt_kws)
sns.scatterplot(data=best_rdpg_left_df, label="RDPG", **plt_kws)

# a priori
plt_kws = dict(
    linewidth=3,
    y=score_name,
    x="n_params",
    ax=ax[0],
    legend=False,
    marker=ap_marker,
    s=ap_size,
)
sns.scatterplot(data=ap_dcsbm_left_df, label="priori DCSBM", **plt_kws)
sns.scatterplot(data=ap_ddcsbm_left_df, label="priori dDCSBM", **plt_kws)

# add text labels for the lines
ax[0].annotate("DCSBM", (200, 0.0525), fontsize=marker_label_fontsize, c=cmap[0])
ax[0].annotate("dDCSBM", (500, 0.0485), fontsize=marker_label_fontsize, c=cmap[1])
ax[0].annotate("RDPG", (1350, 0.045), fontsize=marker_label_fontsize, c=cmap[2])

# add text labels and arrows for the a prioris
point = np.array((ap_dcsbm_left_df.loc[0, "n_params"], ap_dcsbm_left_df.loc[0, "mse"]))
plot_arrow(
    ax[0], point, "A priori DCSBM", cmap[3], marker_label_fontsize=marker_label_fontsize
)

point = np.array(
    (ap_ddcsbm_left_df.loc[0, "n_params"], ap_ddcsbm_left_df.loc[0, "mse"])
)
plot_arrow(
    ax[0],
    point,
    "A priori dDCSBM",
    cmap[4],
    marker_label_fontsize=marker_label_fontsize,
)


# labels
ax[0].set_title("Left", fontsize=title_fontsize)
ax[0].set_xscale("log")
ax[0].set_xlabel("# of parameters", fontsize=axis_label_fontsize)
ax[0].set_ylabel("MSE", fontsize=axis_label_fontsize)


# Right
#
#

# model curves
plt_kws = dict(linewidth=0, y=score_name, x="n_params", ax=ax[1], legend=False)
sns.scatterplot(data=best_dcsbm_right_df, label="DCSBM", **plt_kws)
sns.scatterplot(data=best_ddcsbm_right_df, label="dDCSBM", **plt_kws)
sns.scatterplot(data=best_rdpg_right_df, label="RDPG", **plt_kws)

# a priori
plt_kws = dict(
    linewidth=3,
    y=score_name,
    x="n_params",
    ax=ax[1],
    legend=False,
    marker=ap_marker,
    s=ap_size,
)
sns.scatterplot(data=ap_dcsbm_right_df, label="priori DCSBM", **plt_kws)
sns.scatterplot(data=ap_ddcsbm_right_df, label="priori dDCSBM", **plt_kws)

# add text labels for the lines
ax[1].annotate("DCSBM", (200, 0.0525), fontsize=marker_label_fontsize, c=cmap[0])
ax[1].annotate("dDCSBM", (500, 0.0485), fontsize=marker_label_fontsize, c=cmap[1])
ax[1].annotate("RDPG", (1350, 0.045), fontsize=marker_label_fontsize, c=cmap[2])

# add text labels and arrows for the a prioris
point = np.array(
    (ap_dcsbm_right_df.loc[0, "n_params"], ap_dcsbm_right_df.loc[0, "mse"])
)
plot_arrow(
    ax[1], point, "A priori DCSBM", cmap[3], marker_label_fontsize=marker_label_fontsize
)

point = np.array(
    (ap_ddcsbm_right_df.loc[0, "n_params"], ap_ddcsbm_right_df.loc[0, "mse"])
)
plot_arrow(
    ax[1],
    point,
    "A priori dDCSBM",
    cmap[4],
    marker_label_fontsize=marker_label_fontsize,
)

# labels
ax[1].set_title("Right", fontsize=title_fontsize)
ax[1].set_xscale("log")
ax[1].set_xlabel("# of parameters", fontsize=axis_label_fontsize)
xticks = [2.5e2, 1e3, 5e3]
xticklabels = [f"{i:2.1e}" for i in xticks]
xticklabels = [r"$2.5 \times 10^2$", r"$1 \times 10^3$", r"$5 \times 10^3$"]
ax[1].set_xticks(xticks)
ax[1].set_xticklabels(xticklabels)
plt.tight_layout()
# plt.xticks([5e2, 1e3, 5e3])
plt.savefig(
    "./maggot_models/reports/figures/vertex_aware_mse/vertex_aware_mse.pdf",
    format="pdf",
    facecolor="w",
    bbox_inches="tight",
)


#%%
