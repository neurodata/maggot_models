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
#%% [markdown]
# ### Plot the noise observed in SBM model fitting

#%%
# Plotting setup}
plt.style.use("seaborn-white")
sns.set_context("talk", font_scale=1.5)
plt_kws = dict(s=75, linewidth=0, legend="brief")
sbm_cmap = sns.light_palette("purple", as_cmap=True)

# Plot 1
plt.figure(figsize=(22, 12))
sns.scatterplot(
    data=sbm_df,
    x="n_params_gmm",
    y="mse",
    hue="n_block_try",
    size="n_components_try",
    alpha=0.5,
    palette=sbm_cmap,
    **plt_kws,
)
plt.xlabel("# Params (GMM params for SBMs)")
plt.ylabel("MSE")
plt.title(f"Drosophila old MB left, directed ({experiment}:{run})")

# Plot 2
plt.figure(figsize=(20, 10))
sns.scatterplot(
    data=sbm_df,
    x="n_params_gmm",
    y="mse",
    hue="n_components_try",
    palette=sbm_cmap,
    alpha=0.5,
    **plt_kws,
)
plt.xlabel("# Params (GMM params for SBMs)")
plt.ylabel("MSE")
plt.title(f"Drosophila old MB left, directed ({experiment}:{run})")

# Plot 3
plt.figure(figsize=(22, 12))
sns.scatterplot(
    data=sbm_df,
    x="n_params_sbm",
    y="mse",
    hue="n_block_try",
    size="n_components_try",
    alpha=0.5,
    palette=sbm_cmap,
    **plt_kws,
)
plt.xlabel("# Params (SBM params)")
plt.ylabel("MSE")
plt.title(f"Drosophila old MB left, directed ({experiment}:{run})")

# Plot 4
plt.figure(figsize=(20, 10))
sns.scatterplot(
    data=sbm_df,
    x="n_params_sbm",
    y="mse",
    hue="n_components_try",
    palette=sbm_cmap,
    alpha=0.5,
    **plt_kws,
)
plt.xlabel("# Params (SBM params)")
plt.ylabel("MSE")
plt.title(f"Drosophila old MB left, directed ({experiment}:{run})")


#%%
best_sbm_df = utils.get_best_df3(sbm_df)

#%% GMM params - with hue
plt.figure(figsize=(22, 12))
plt_kws = dict(s=75, linewidth=0, legend="brief")


cmap = sns.light_palette("purple", as_cmap=True)
sns.scatterplot(
    data=best_sbm_df, x="n_params_gmm", y="mse", hue="rank_try", palette=cmap, **plt_kws
)


plt.xlabel("# Params (GMM params for SBMs)")
plt.ylabel("MSE")
plt.title(f"Drosophila old MB left, directed ({experiment}:{run})")

#%% SBM params - with hue
plt.figure(figsize=(22, 12))
plt_kws = dict(s=75, linewidth=0, legend="brief")


cmap = sns.light_palette("purple", as_cmap=True)
best_sbm_df["rank_proportion"] = best_sbm_df["rank_try"] / best_sbm_df["n_block_try"]
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

plt.figure(figsize=(22, 12))
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

plt.figure(figsize=(22, 12))

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

#%% GMM params - no hue
sns.set_palette("Set1")

plt.figure(figsize=(22, 12))
plt_kws = dict(s=75, linewidth=0, legend="brief")


sns.scatterplot(
    data=best_sbm_df,
    x="n_params_gmm",
    y="mse",
    palette=cmap,
    label="GraspyGMM",
    **plt_kws,
)

s = sns.scatterplot(
    data=tsbm_df, x="n_params_gmm", y="mse", palette=cmap, label="AutoGMM", **plt_kws
)

plt.xlabel("# Params (GMM params for SBMs)")
plt.ylabel("MSE")
plt.title(f"Drosophila old MB left, directed ({experiment}:{run})")

#%% SBM params - no hue
plt.figure(figsize=(22, 12))
plt_kws = dict(s=75, linewidth=0, legend="brief")

sns.scatterplot(
    data=best_sbm_df, x="n_params_sbm", y="mse", label="GraspyGMM", **plt_kws
)


s = sns.scatterplot(data=tsbm_df, x="n_params_sbm", y="mse", label="AutoGMM", **plt_kws)

plt.xlabel("# Params (SBM params for SBMs)")
plt.ylabel("MSE")
plt.title(f"Drosophila old MB left, directed ({experiment}:{run})")


#%%
best_tsbm_df = utils.get_best_df2(tsbm_df)
#%% GMM params - no hue
sns.set_palette("Set1")

plt.figure(figsize=(22, 12))
plt_kws = dict(s=75, linewidth=0, legend="brief")


sns.scatterplot(
    data=best_sbm_df,
    x="n_params_gmm",
    y="mse",
    palette=cmap,
    label="GraspyGMM",
    **plt_kws,
)

s = sns.scatterplot(
    data=best_tsbm_df,
    x="n_params_gmm",
    y="mse",
    palette=cmap,
    label="AutoGMM",
    **plt_kws,
)

plt.xlabel("# Params (GMM params for SBMs)")
plt.ylabel("MSE")
plt.title(f"Drosophila old MB left, directed ({experiment}:{run})")

#%% SBM params - no hue
plt.figure(figsize=(22, 12))
plt_kws = dict(s=75, linewidth=0, legend="brief")

sns.scatterplot(
    data=best_sbm_df, x="n_params_sbm", y="mse", label="GraspyGMM", **plt_kws
)


s = sns.scatterplot(
    data=best_tsbm_df, x="n_params_sbm", y="mse", label="AutoGMM", **plt_kws
)

plt.xlabel("# Params (SBM params for SBMs)")
plt.ylabel("MSE")
plt.title(f"Drosophila old MB left, directed ({experiment}:{run})")


#%%
