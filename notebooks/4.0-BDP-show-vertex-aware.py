#%%
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from IPython import get_ipython  # just to decieve flake8

import src.utils as utils

get_ipython().run_line_magic("autoreload", "2")

get_ipython().run_line_magic("matplotlib", "inline")
os.getcwd()

#%% [markdown]
# ### Choose experiment, print out configurations

#%% Load DCSBM
base_path = "./maggot_models/models/runs/"
experiment = "fit_dcsbm"
run = 7
config = utils.load_config(base_path, experiment, run)
dcsbm_df = utils.load_pickle(base_path, experiment, run, "dcsbm_out_df")

#%% Load dDCSBM
base_path = "./maggot_models/models/runs/"
experiment = "fit_ddcsbm"
run = 2
config = utils.load_config(base_path, experiment, run)
ddcsbm_df = utils.load_pickle(base_path, experiment, run, "ddcsbm_out_df")

#%% Load RDPG
base_path = "./maggot_models/models/runs/"
experiment = "fit_rdpg"
run = 2
config = utils.load_config(base_path, experiment, run)
rdpg_df = utils.load_pickle(base_path, experiment, run, "rdpg_out_df")


#%%
def get_best(df, param_name="param_n_components", score_name="mse"):
    param_range = np.unique(df[param_name].values)
    best_rows = []
    for param_value in param_range:
        temp_df = df[df[param_name] == param_value]
        ind = temp_df[score_name].idxmin()  # this is the metric we are choosing on
        best_rows.append(temp_df.loc[ind, :])
    return pd.DataFrame(best_rows)


best_dcsbm_df = get_best(dcsbm_df, param_name="param_n_blocks")
best_ddcsbm_df = get_best(ddcsbm_df, param_name="param_n_blocks")
best_rdpg_df = get_best(rdpg_df, param_name="param_n_components")

#%% Settings for all plots
plt.style.use("seaborn-white")
sns.set_context("talk")
sns.set_palette("Set1")


#%% Figure - show RDPG, DCSBM bests
plt.figure(figsize=(20, 10))
plt_kws = dict(linewidth=0, y="mse", x="n_params")
sns.scatterplot(data=best_dcsbm_df, label="DCSBM", **plt_kws)
sns.scatterplot(data=best_ddcsbm_df, label="dDCSBM", **plt_kws)
sns.scatterplot(data=best_rdpg_df, label="RDPG", **plt_kws)
plt.legend()

#%%
plt.figure(figsize=(20, 10))
plt_kws = dict(linewidth=0, y="likelihood", x="n_params")
sns.scatterplot(data=best_dcsbm_df, label="DCSBM", **plt_kws)
sns.scatterplot(data=best_ddcsbm_df, label="dDCSBM", **plt_kws)
sns.scatterplot(data=best_rdpg_df, label="RDPG", **plt_kws)
plt.legend()

#%% Figure - observe the effect of regularization on RDPG fitting
plot_df = rdpg_df[rdpg_df["param_plus_c_weight"] == 0]
plt.figure(figsize=(20, 10))
sns.scatterplot(
    data=plot_df,
    x="n_params",
    y="mse",
    hue="param_diag_aug_weight",
    size="param_diag_aug_weight",
    alpha=0.5,
    linewidth=0,
)


#%%
