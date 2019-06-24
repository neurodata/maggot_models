#%%
import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from IPython import get_ipython  # just to decieve flake8

import src.utils as utils

get_ipython().run_line_magic("autoreload", "2")

get_ipython().run_line_magic("matplotlib", "inline")
os.getcwd()


#%% [markdown]
# ### Choose experiment, print out configurations

#%%
base_path = "./maggot_models/models/runs/"
experiment = "drosophila-5-rdpg-sbm"
run = 4
config = utils.load_config(base_path, experiment, run)
sbm_df = utils.load_pickle(base_path, experiment, run, "sbm_master_df")
sbm_df = sbm_df.apply(pd.to_numeric)


def get_best_df(input_df):
    """super hard coded right now (e.g. column names)
    
    Parameters
    ----------
    df : dataframe
        [description]
    
    Returns
    -------
    [type]
        [description]
    """
    param_df = input_df[input_df["sim_ind"] == 0]
    labels = ["n_block_try", "rank_try", "mse"]
    param_df = param_df.loc[:, labels]
    param_df["best_sim"] = 0
    param_df["best_ind"] = 0
    for i in range(50):
        df = input_df[input_df["sim_ind"] == i]
        for j, row in df.iterrows():
            temp_df = param_df.loc[
                (param_df[labels[0]] == row[labels[0]])
                & (param_df[labels[1]] == row[labels[1]])
            ]
            ind = temp_df.index
            if row["mse"] <= param_df.loc[ind, "mse"].values[0]:
                param_df.loc[ind, "mse"] = row["mse"]
                param_df.loc[ind, "best_sim"] = row["sim_ind"]
                param_df.loc[ind, "best_ind"] = j
    best_df = input_df.loc[param_df["best_ind"].values, :]
    return best_df


best_sbm_df = get_best_df(sbm_df)
best_sbm_df["rank_proportion"] = best_sbm_df["rank_try"] / best_sbm_df["n_block_try"]

#%%
# Plotting setup}
plt.style.use("seaborn-white")
sns.set_context("talk", font_scale=1.5)
plt_kws = dict(s=75, linewidth=0, legend="brief")
figsize = (22, 12)
cmap = sns.light_palette("purple", as_cmap=True)
save_dir = Path("./maggot_models/reports/figures/vertex_agnostic_mse")
#%% Figures - rank SBMs


plt.figure(figsize=figsize)
sns.scatterplot(
    data=best_sbm_df,
    x="n_params_sbm",
    y="mse",
    size="n_block_try",
    hue="rank_try",
    palette=cmap,
    **plt_kws,
)
plt.xlabel("# Params (SBM params for SBMs)")
plt.ylabel("MSE")
plt.title(f"Drosophila old MB left, directed ({experiment}:{run})")
plt.savefig(save_dir / "rank_sbm_1.pdf", format="pdf", facecolor="w")


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

#%%
