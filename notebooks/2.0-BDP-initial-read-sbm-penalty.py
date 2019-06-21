#%%
import os
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
from IPython import get_ipython  # just to decieve flake8
import ijson
import src.utils as utils
import numpy as np
import pandas as pd

import seaborn as sns

base_path = "./maggot_models/simulations/runs/"
experiment = "sbm_rss_lik_sim"
run = 8
config = utils.load_config(base_path, experiment, run)
sbm_df = utils.load_pickle(base_path, experiment, run, "master_out_df")
# sbm_df = pd.to_numeric(sbm_df["n_verts"])
sbm_df.head()

#%%

#%% [markdown]
## Compute the penalty on the dataframe
if config["directed"]:
    # n^2 - n (no loops)
    sbm_df["n_obs"] = sbm_df["n_verts"] * sbm_df["n_verts"] - sbm_df["n_verts"]
else:
    # n (n - 1) / 2
    sbm_df["n_obs"] = sbm_df["n_verts"] * (sbm_df["n_verts"] - 1) / 2
sbm_df["n_obs"] = pd.to_numeric(sbm_df["n_obs"])

sbm_df["penalized_rss"] = -sbm_df["n_obs"] * np.log(sbm_df["mse"])
sbm_df["penalized_rss"] += -sbm_df["n_params_gmm"] * np.log(sbm_df["n_obs"].values) ** 2

sbm_df["n_verts"] = pd.to_numeric(sbm_df["n_verts"])
#%%
plt.style.use("seaborn-white")
sns.set_context("talk")
sns.set_palette("Set1")
n_blocks_range = config["n_blocks_range"]
n_components_try_range = config["n_components_try_range"]
for n_blocks in n_blocks_range:
    true_k_sbm_df = sbm_df[sbm_df["n_blocks"] == n_blocks]
    correct_embed_df = true_k_sbm_df[true_k_sbm_df["n_components_try"] == n_blocks]
    plt.figure(figsize=(20, 10))
    sns.scatterplot(
        data=correct_embed_df,
        x="n_block_try",
        y="penalized_rss",
        hue="n_verts",
        legend=False,
    )
    plt.title(f"True K = {n_blocks}")
    plt.show()


#%%
for n_blocks in n_blocks_range:
    true_k_sbm_df = sbm_df[sbm_df["n_blocks"] == n_blocks]
    correct_embed_df = true_k_sbm_df[true_k_sbm_df["n_components_try"] == n_blocks]
    plt.figure(figsize=(20, 10))
    sns.scatterplot(
        data=correct_embed_df, x="n_block_try", y="mse", hue="n_verts", legend="brief"
    )
    plt.axvline(x=n_blocks, c="k", alpha=0.5, linestyle="--")
    plt.title(f"True K = {n_blocks}")
    plt.show()

#%%
n_verts_range = config["n_verts_range"]
for n_verts in n_verts_range:
    for n_blocks in n_blocks_range:
        true_k_sbm_df = sbm_df[sbm_df["n_blocks"] == n_blocks]
        true_k_sbm_df = true_k_sbm_df[true_k_sbm_df["n_verts"] == n_verts]
        correct_embed_df = true_k_sbm_df[true_k_sbm_df["n_components_try"] == n_blocks]
        plt.figure(figsize=(20, 10))
        sns.scatterplot(data=correct_embed_df, x="n_block_try", y="penalized_rss")
        plt.axvline(x=n_blocks, c="k", alpha=0.5, linestyle="--")
        plt.title(f"True K = {n_blocks}")
        plt.show()
