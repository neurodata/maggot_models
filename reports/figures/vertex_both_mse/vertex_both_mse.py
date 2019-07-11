#%%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import src.utils as utils
from matplotlib import rc

# Vertex agnostic
#
#

# Load SBM
base_path = "./maggot_models/models/runs/"
experiment = "fit_sbm"
run = 1
config = utils.load_config(base_path, experiment, run)
sbm_left_df = utils.load_pickle(base_path, experiment, run, "sbm_left_df")
sbm_right_df = utils.load_pickle(base_path, experiment, run, "sbm_right_df")

# Load a priori
experiment = "fit_a_priori"
run = 1
config = utils.load_config(base_path, experiment, run)
ap_sbm_left_df = utils.load_pickle(base_path, experiment, run, "sbm_left_df")
ap_sbm_right_df = utils.load_pickle(base_path, experiment, run, "sbm_right_df")

# Load opposites
experiment = "fit_opposite_a_priori"
run = 1
config = utils.load_config(base_path, experiment, run)
right_pred_sbm_df = utils.load_pickle(base_path, experiment, run, "right_pred_sbm_df")
left_pred_sbm_df = utils.load_pickle(base_path, experiment, run, "left_pred_sbm_df")

# Vertex aware
#
#

# Load DCSBM
base_path = "./maggot_models/models/runs/"
experiment = "fit_dcsbm"
run = 8
config = utils.load_config(base_path, experiment, run)
dcsbm_left_df = utils.load_pickle(base_path, experiment, run, "dcsbm_left_df")
dcsbm_right_df = utils.load_pickle(base_path, experiment, run, "dcsbm_right_df")

# Load dDCSBM
experiment = "fit_ddcsbm"
run = 3
config = utils.load_config(base_path, experiment, run)
ddcsbm_left_df = utils.load_pickle(base_path, experiment, run, "ddcsbm_left_df")
ddcsbm_right_df = utils.load_pickle(base_path, experiment, run, "ddcsbm_right_df")

# Load RDPG
experiment = "fit_rdpg"
run = 3
config = utils.load_config(base_path, experiment, run)
rdpg_left_df = utils.load_pickle(base_path, experiment, run, "rdpg_left_df")
rdpg_right_df = utils.load_pickle(base_path, experiment, run, "rdpg_right_df")

# Load a priori
experiment = "fit_a_priori"
run = 1
config = utils.load_config(base_path, experiment, run)
ap_dcsbm_left_df = utils.load_pickle(base_path, experiment, run, "dcsbm_left_df")
ap_dcsbm_right_df = utils.load_pickle(base_path, experiment, run, "dcsbm_right_df")

ap_ddcsbm_left_df = utils.load_pickle(base_path, experiment, run, "ddcsbm_left_df")
ap_ddcsbm_right_df = utils.load_pickle(base_path, experiment, run, "ddcsbm_right_df")

#%% Get best parameters


def get_best(df, param_name="param_n_components", score_name="mse"):
    param_range = np.unique(df[param_name].values)
    best_rows = []
    for param_value in param_range:
        temp_df = df[df[param_name] == param_value]
        ind = temp_df[score_name].idxmin()  # this is the metric we are choosing on
        best_rows.append(temp_df.loc[ind, :])
    return pd.DataFrame(best_rows)


score_name = "mse"
best_sbm_left_df = get_best(
    sbm_left_df, param_name="param_n_blocks", score_name=score_name
)
best_sbm_right_df = get_best(
    sbm_right_df, param_name="param_n_blocks", score_name=score_name
)
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

#%% Plot


plt.style.use("seaborn-white")
sns.set_context("talk", font_scale=1.5)

# build a colormap
base_cmap = sns.color_palette("Set1", n_colors=4, desat=1)
base_cmap = []
base_desat = 0.9
n_colors = 3
red = sns.color_palette("Reds", n_colors=n_colors, desat=base_desat)[-1]
base_cmap.append(red)
blue = sns.color_palette("Blues", n_colors=n_colors, desat=base_desat)[-1]
base_cmap.append(blue)
green = sns.color_palette("Greens", n_colors=n_colors, desat=base_desat)[-1]
base_cmap.append(green)
purp = sns.color_palette("Purples", n_colors=n_colors, desat=base_desat)[-1]
base_cmap.append(purp)

ap_cmap = []
n_colors = 3
step_down = 2
light_red = sns.color_palette("Reds", n_colors=n_colors, desat=base_desat)[-step_down]
ap_cmap.append(light_red)
light_blue = sns.color_palette("Blues", n_colors=n_colors, desat=base_desat)[-step_down]
ap_cmap.append(light_blue)
light_green = sns.color_palette("Greens", n_colors=n_colors, desat=base_desat)[
    -step_down
]
ap_cmap.append(light_green)

cmap = base_cmap + ap_cmap
sns.palplot(cmap)
set1 = sns.color_palette("Set1", n_colors=4, desat=base_desat)
sns.palplot(set1)

#%%
# sns.palplot(base_cmap)
# ap_cmap = sns.color_palette("Set1", desat=0.3, n_colors=2)
# sns.palplot(ap_cmap)
# cmap = sns.color_palette(["#de2d26", "#3182bd", "#31a354", "#fb6a4a", "#6baed6"])


# settings
title_fontsize = 40
axis_label_fontsize = 30
marker_label_fontsize = 25
ap_marker = "x"
ap_size = 250
ap_linewidth = 3
# start figure
fig, ax = plt.subplots(2, 2, figsize=(20, 20))


# vertex aware
aware_xticks = [2e2, 2e3]
aware_xticklabels = [r"$2 \times 10^2$", r"$2 \times 10^3$"]
aware_ylim = (0.035, 0.1)
aware_xlim = (1e2, 1e4)


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


# Vertex agnostic left
#
#
current_ax = ax[0, 0]

# model curves
plt_kws = dict(
    linewidth=0, y=score_name, x="n_params", ax=current_ax, legend=False, c=cmap[0]
)
sns.set_palette(cmap)
sns.scatterplot(data=best_sbm_left_df, label="SBM", **plt_kws)
current_ax.annotate("SBM", (20, 0.058), fontsize=marker_label_fontsize, c=cmap[0])

# a priori
plt_kws = dict(
    linewidth=ap_linewidth,
    y=score_name,
    x="n_params",
    ax=current_ax,
    marker=ap_marker,
    s=ap_size,
    legend=False,
    c=cmap[4],
)
sns.scatterplot(data=ap_sbm_left_df, label="priori SBM", **plt_kws)
point = (ap_sbm_left_df.loc[0, "n_params"], ap_sbm_left_df.loc[0, "mse"])
plot_arrow(
    current_ax,
    point,
    "A priori SBM - left",
    cmap[4],
    marker_label_fontsize=marker_label_fontsize,
)

sns.scatterplot(data=left_pred_sbm_df, **plt_kws)
point = (left_pred_sbm_df.loc[0, "n_params"], ap_sbm_left_df.loc[0, "mse"])
plot_arrow(
    current_ax,
    point,
    "A priori SBM - right",
    cmap[4],
    marker_label_fontsize=marker_label_fontsize,
)


current_ax.set_title("Vertex agnostic - left", fontsize=title_fontsize)
current_ax.set_xscale("log")
current_ax.set_xlabel("# of parameters", fontsize=axis_label_fontsize)
current_ax.set_ylabel("MSE", fontsize=axis_label_fontsize)

# Vertex agnostic right
#
#
current_ax = ax[0, 1]
# model curves
plt_kws = dict(
    linewidth=0, y=score_name, x="n_params", ax=current_ax, legend=False, c=cmap[0]
)
sns.scatterplot(data=best_sbm_right_df, label="SBM", **plt_kws)
current_ax.annotate("SBM", (20, 0.06), fontsize=marker_label_fontsize, c=cmap[0])

# a priori
plt_kws = dict(
    linewidth=ap_linewidth,
    y=score_name,
    x="n_params",
    ax=current_ax,
    marker=ap_marker,
    s=ap_size,
    legend=False,
    c=cmap[4],
)
sns.scatterplot(data=ap_sbm_right_df, label="priori SBM", **plt_kws)
point = (ap_sbm_right_df.loc[0, "n_params"], ap_sbm_right_df.loc[0, "mse"])
plot_arrow(
    current_ax,
    point,
    "A priori SBM - right",
    cmap[4],
    marker_label_fontsize=marker_label_fontsize,
)

# labels
current_ax.set_title("Vertex agnostic - right", fontsize=title_fontsize)
current_ax.set_xscale("log")
current_ax.set_xlabel("# of parameters", fontsize=axis_label_fontsize)
current_ax.set_ylabel("MSE", fontsize=axis_label_fontsize)


# Vertex aware left
#
#
sns.set_palette(cmap[1:])

current_ax = ax[1, 0]

# model curves
plt_kws = dict(linewidth=0, y=score_name, x="n_params", ax=current_ax, legend=False)
sns.scatterplot(data=best_dcsbm_left_df, label="DCSBM", c=cmap[1], **plt_kws)
sns.scatterplot(data=best_ddcsbm_left_df, label="dDCSBM", c=cmap[2], **plt_kws)
sns.scatterplot(data=best_rdpg_left_df, label="RDPG", c=cmap[3], **plt_kws)

# a priori
plt_kws = dict(
    linewidth=ap_linewidth,
    y=score_name,
    x="n_params",
    ax=current_ax,
    legend=False,
    marker=ap_marker,
    s=ap_size,
    c=cmap[5],
)
sns.scatterplot(data=ap_dcsbm_left_df, label="priori DCSBM", **plt_kws)
plt_kws = dict(
    linewidth=ap_linewidth,
    y=score_name,
    x="n_params",
    ax=current_ax,
    legend=False,
    marker=ap_marker,
    s=ap_size,
    c=cmap[6],
)
sns.scatterplot(data=ap_ddcsbm_left_df, label="priori dDCSBM", **plt_kws)

# add text labels for the lines
current_ax.annotate("DCSBM", (200, 0.0525), fontsize=marker_label_fontsize, c=cmap[1])
current_ax.annotate("dDCSBM", (500, 0.0485), fontsize=marker_label_fontsize, c=cmap[2])
current_ax.annotate("RDPG", (1350, 0.045), fontsize=marker_label_fontsize, c=cmap[3])

# add text labels and arrows for the a prioris
point = np.array((ap_dcsbm_left_df.loc[0, "n_params"], ap_dcsbm_left_df.loc[0, "mse"]))
plot_arrow(
    current_ax,
    point,
    "A priori DCSBM",
    cmap[5],
    marker_label_fontsize=marker_label_fontsize,
)

point = np.array(
    (ap_ddcsbm_left_df.loc[0, "n_params"], ap_ddcsbm_left_df.loc[0, "mse"])
)
plot_arrow(
    current_ax,
    point,
    "A priori dDCSBM",
    cmap[6],
    marker_label_fontsize=marker_label_fontsize,
)


# labels
current_ax.set_title("Vertex aware - left", fontsize=title_fontsize)
current_ax.set_xscale("log")
current_ax.set_xlabel("# of parameters", fontsize=axis_label_fontsize)
current_ax.set_ylabel("MSE", fontsize=axis_label_fontsize)

current_ax.set_xticks(aware_xticks)
current_ax.set_xticklabels(aware_xticklabels)
current_ax.set_xlim(*aware_xlim)
current_ax.set_ylim(*aware_ylim)

# Right
#
#
current_ax = ax[1, 1]


# model curves
plt_kws = dict(linewidth=0, y=score_name, x="n_params", ax=current_ax, legend=False)
sns.scatterplot(data=best_dcsbm_right_df, label="DCSBM", c=cmap[1], **plt_kws)
sns.scatterplot(data=best_ddcsbm_right_df, label="dDCSBM", c=cmap[2], **plt_kws)
sns.scatterplot(data=best_rdpg_right_df, label="RDPG", c=cmap[3], **plt_kws)

# a priori
plt_kws = dict(
    linewidth=ap_linewidth,
    y=score_name,
    x="n_params",
    ax=current_ax,
    legend=False,
    marker=ap_marker,
    s=ap_size,
    c=cmap[5],
)
sns.scatterplot(data=ap_dcsbm_right_df, label="priori DCSBM", **plt_kws)
plt_kws = dict(
    linewidth=ap_linewidth,
    y=score_name,
    x="n_params",
    ax=current_ax,
    legend=False,
    marker=ap_marker,
    s=ap_size,
    c=cmap[6],
)
sns.scatterplot(data=ap_ddcsbm_right_df, label="priori dDCSBM", **plt_kws)

# add text labels for the lines
current_ax.annotate("DCSBM", (200, 0.0525), fontsize=marker_label_fontsize, c=cmap[1])
current_ax.annotate("dDCSBM", (500, 0.0485), fontsize=marker_label_fontsize, c=cmap[2])
current_ax.annotate("RDPG", (1350, 0.045), fontsize=marker_label_fontsize, c=cmap[3])

# add text labels and arrows for the a prioris
point = np.array(
    (ap_dcsbm_right_df.loc[0, "n_params"], ap_dcsbm_right_df.loc[0, "mse"])
)
plot_arrow(
    current_ax,
    point,
    "A priori DCSBM",
    cmap[5],
    marker_label_fontsize=marker_label_fontsize,
)

point = np.array(
    (ap_ddcsbm_right_df.loc[0, "n_params"], ap_ddcsbm_right_df.loc[0, "mse"])
)
plot_arrow(
    current_ax,
    point,
    "A priori dDCSBM",
    cmap[6],
    marker_label_fontsize=marker_label_fontsize,
)

# labels
current_ax.set_title("Vertex aware - right", fontsize=title_fontsize)
current_ax.set_xscale("log")
current_ax.set_xlabel("# of parameters", fontsize=axis_label_fontsize)
current_ax.set_ylabel("MSE", fontsize=axis_label_fontsize)

# globals

current_ax.set_xticks(aware_xticks)
current_ax.set_xticklabels(aware_xticklabels)
current_ax.set_xlim(*aware_xlim)
current_ax.set_ylim(*aware_ylim)
plt.tight_layout()
# plt.xticks([5e2, 1e3, 5e3])
plt.savefig(
    "./maggot_models/reports/figures/vertex_both_mse/vertex_both_mse.pdf",
    format="pdf",
    facecolor="w",
    bbox_inches="tight",
)

#%% JUNK
# #%% Load coSBM
# base_path = "./maggot_models/models/runs/"
# experiment = "fit_cosbm"
# run = 1
# config = utils.load_config(base_path, experiment, run)
# cosbm_left_df = utils.load_pickle(base_path, experiment, run, "cosbm_left_df")
# cosbm_right_df = utils.load_pickle(base_path, experiment, run, "cosbm_right_df")


# best_cosbm_left_df = get_best(
#     cosbm_left_df, param_name="param_n_blocks", score_name=score_name
# )
# best_cosbm_right_df = get_best(
#     cosbm_right_df, param_name="param_n_blocks", score_name=score_name
# )


# #%%
# # build a colormap
# base_cmap = sns.color_palette("Set1", n_colors=1, desat=1)
# sns.palplot(base_cmap)
# ap_cmap = sns.color_palette("Set1", desat=0.3, n_colors=2)
# sns.palplot(ap_cmap)
# custom_cmap = sns.color_palette(["#de2d26", "#fb6a4a"])
# # custom_cmap = base_cmap + ap_cmap
# sns.set_palette(custom_cmap)
# #%%
# title_fontsize = 40
# axis_label_fontsize = 30
# ap_marker = "*"
# ap_size = 200
# #%% Figure - show SBM bests
# # Left
# fig, ax = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(20, 10))


# #%%
