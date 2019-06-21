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
experiment = "drosophila-3-rdpg-sbm"
run = 13
sbm_df, _, tsbm_df = utils.load_run(base_path, experiment, run)


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
sns.scatterplot(
    data=rdpg_df, x="n_params", y="mse", hue="RDPG", palette=rdpg_cmap, **plt_kws
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
sns.scatterplot(
    data=rdpg_df, x="n_params", y="mse", hue="RDPG", palette=rdpg_cmap, **plt_kws
)
plt.xlabel("# Params (GMM params for SBMs)")
plt.ylabel("MSE")
plt.title(f"Drosophila old MB left, directed ({experiment}:{run})")

# # Plot 3
# plt.figure(figsize=(20,10))
# sns.scatterplot(data=sbm_df, x="n_params_sbm", y="mse", hue="n_components_try", **plt_kws, alpha=0.5)
# sns.scatterplot(data=rdpg_df, x="n_params", y="mse", **plt_kws)
# plt.xlabel("# Params (SBM params for SBMs)")
# plt.ylabel("MSE")
# plt.title(f"Drosophila old MB left, directed ({experiment}:{run})");

#%% [markdown]
# ### Get the best MSE SBM model fitting for each parameter set

#%%


#%%
best_sbm_df = utils.get_best_df(sbm_df)
best_tsbm_df = utils.get_best_df(tsbm_df)
plt.figure(figsize=(22, 12))
cmap = sns.light_palette("purple", as_cmap=True)

sns.scatterplot(
    data=best_sbm_df,
    x="n_params_gmm",
    y="mse",
    hue="n_block_try",
    size="n_components_try",
    palette=cmap,
    **plt_kws,
)

cmap = sns.xkcd_palette(["grass green"])
s = sns.scatterplot(
    data=rdpg_df, x="n_params", y="mse", hue="RDPG", palette=cmap, **plt_kws
)

plt.xlabel("# Params (GMM params for SBMs)")
plt.ylabel("MSE")
plt.title(f"Drosophila old MB left, directed ({experiment}:{run})")


#%%
plt.figure(figsize=(22, 12))
plt_kws = dict(s=75, linewidth=0, legend=False)


cmap = sns.light_palette("purple", as_cmap=True)
sns.scatterplot(
    data=best_sbm_df,
    x="n_params_gmm",
    y="mse",
    hue="n_components_try",
    size="n_block_try",
    palette=cmap,
    **plt_kws,
)


cmap = sns.light_palette("teal", as_cmap=True)
s = sns.scatterplot(
    data=best_tsbm_df,
    x="n_params_gmm",
    y="mse",
    hue="n_components_try",
    size="n_block_try",
    palette=cmap,
    **plt_kws,
)


cmap = sns.xkcd_palette(["grass green"])
s = sns.scatterplot(
    data=rdpg_df, x="n_params", y="mse", hue="RDPG", palette=cmap, **plt_kws
)

plt.xlabel("# Params (GMM params for SBMs)")
plt.ylabel("MSE")
plt.title(f"Drosophila old MB left, directed ({experiment}:{run})")

