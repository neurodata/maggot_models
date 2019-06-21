#%%
import os
from pathlib import Path

import matplotlib.pyplot as plt
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
experiment = "drosophila-4-rdpg-sbm"
run = 1
config = utils.load_config(base_path, experiment, run)
sbm_df = utils.load_pickle(base_path, experiment, run, "sbm_master_df")
tsbm_df = utils.load_pickle(base_path, experiment, run, "tsbm_df")

#%% [markdown]
# ### Plot the noise observed in SBM model fitting

#%%
# Plotting setup}
plt.style.use("seaborn-white")
sns.set_context("talk", font_scale=1.5)
plt_kws = dict(s=75, linewidth=0, legend="brief")
sbm_cmap = sns.light_palette("purple", as_cmap=True)
rdpg_cmap = sns.xkcd_palette(["grass green"])

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


best_sbm_df = utils.get_best_df2(sbm_df)

#%%
plt.figure(figsize=(22, 12))

cmap = sns.light_palette("purple", as_cmap=True)

sns.scatterplot(
    data=best_sbm_df,
    x="n_params_sbm",
    y="mse",
    hue="n_block_try",
    size="n_components_try",
    palette=cmap,
    **plt_kws,
)

plt.xlabel("# Params (GMM params)")
plt.ylabel("MSE")

cmap = sns.light_palette("purple", as_cmap=True)
sns.scatterplot(
    data=tsbm_df,
    x="n_params_sbm",
    y="mse",
    hue="n_block_try",
    size="n_components_try",
    palette=cmap,
    **plt_kws,
)

plt.xlabel("# Params (SBM params)")
plt.ylabel("MSE")
plt.title(f"Drosophila old MB left, directed ({experiment}:{run})")


#%% GMM params
plt.figure(figsize=(22, 12))
plt_kws = dict(s=75, linewidth=0, legend="brief")


cmap = sns.light_palette("purple", as_cmap=True)
sns.scatterplot(
    data=best_sbm_df,
    x="n_params_gmm",
    y="mse",
    hue="n_components_try",
    palette=cmap,
    **plt_kws,
)


cmap = sns.light_palette("teal", as_cmap=True)
s = sns.scatterplot(
    data=tsbm_df,
    x="n_params_gmm",
    y="mse",
    hue="n_components_try",
    palette=cmap,
    **plt_kws,
)

plt.xlabel("# Params (GMM params for SBMs)")
plt.ylabel("MSE")
plt.title(f"Drosophila old MB left, directed ({experiment}:{run})")

#%% SBM params
plt.figure(figsize=(22, 12))
plt_kws = dict(s=75, linewidth=0, legend="brief")


cmap = sns.light_palette("purple", as_cmap=True)
sns.scatterplot(
    data=best_sbm_df,
    x="n_params_sbm",
    y="mse",
    hue="n_components_try",
    palette=cmap,
    **plt_kws,
)


cmap = sns.light_palette("teal", as_cmap=True)
s = sns.scatterplot(
    data=tsbm_df,
    x="n_params_sbm",
    y="mse",
    hue="n_components_try",
    palette=cmap,
    **plt_kws,
)

plt.xlabel("# Params (SBM params for SBMs)")
plt.ylabel("MSE")
plt.title(f"Drosophila old MB left, directed ({experiment}:{run})")

