#%% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataScience.changeDirOnImportExport setting
# ms-python.python added
import os

try:
    os.chdir(os.path.join(os.getcwd(), "maggot_models/notebooks"))
    print(os.getcwd())
except:
    pass

#%%
from pathlib import Path
import os

get_ipython().run_line_magic("autoreload", "2")
import src.utils as utils
import seaborn as sns
import matplotlib.pyplot as plt

get_ipython().run_line_magic("matplotlib", "inline")
os.getcwd()

#%% [markdown]
# ### Choose experiment, print out configurations

#%%
experiment = "drosophila-3-rdpg-sbm"
run = 6
exp_path = Path(f"../models/runs/{experiment}/{run}")
run_path = exp_path / "run.json"
config_path = exp_path / "config.json"

config = utils.get_json(config_path)
print(f"Experiment: {experiment}")
print(f"Run: {run}")
print(f"Path: {run_path}")
print()
print("Experiment configuration:")
print()
for key, value in config.items():
    if not key == "__doc__":
        print(key)
        print(value)
        print()

dfs = utils.run_to_df(run_path)
sbm_df = dfs[0]
rdpg_df = dfs[1]
tsbm_df = dfs[2]

rdpg_df["RDPG"] = "RDPG"

#%% [markdown]
# ### Plot the noise observed in SBM model fitting

#%%
# Plotting setup
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
param_df = sbm_df[sbm_df["sim_ind"] == 0]
labels = ["n_block_try", "n_components_try", "mse"]
param_df = param_df.loc[:, labels]
param_df["best_sim"] = 0
param_df["best_ind"] = 0
for i in range(50):
    df = sbm_df[sbm_df["sim_ind"] == i]
    for j, row in df.iterrows():
        params = row[labels]
        p_df = param_df.loc[
            (param_df[labels[0]] == row[labels[0]])
            & (param_df[labels[1]] == row[labels[1]])
        ]
        ind = p_df.index
        if row["mse"] <= param_df.loc[ind, "mse"].values[0]:
            param_df.loc[ind, "mse"] = row["mse"]
            param_df.loc[ind, "best_sim"] = row["sim_ind"]
            param_df.loc[ind, "best_ind"] = j
best_sbm_df = sbm_df.loc[param_df["best_ind"].values, :]


#%%
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

leg = s.axes.get_legend()
leg.get_texts()[0].set_text("SBM: K, best of 50")
leg.get_texts()[6].set_text("SBM: d, best of 50")
leg.get_texts()[11].set_text("RDPG: directed")
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
    data=tsbm_df,
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

# leg = s.axes.get_legend()
# leg.get_texts()[0].set_text("SBM: d, best of 50")
# leg.get_texts()[5].set_text("SBM: K, best of 50")
# leg.get_texts()[11].set_text("RDPG: directed")
plt.xlabel("# Params (GMM params for SBMs)")
plt.ylabel("MSE")
plt.title(f"Drosophila old MB left, directed ({experiment}:{run})")

#%%
plt.figure(figsize=(22, 12))

cmap = sns.light_palette("purple", as_cmap=True)
sns.scatterplot(
    data=best_sbm_df,
    x="n_params_gmm",
    y="score",
    hue="n_block_try",
    size="n_components_try",
    palette=cmap,
    **plt_kws,
)

cmap = sns.xkcd_palette(["grass green"])
s = sns.scatterplot(
    data=rdpg_df, x="n_params", y="score", hue="RDPG", palette=cmap, **plt_kws
)

leg = s.axes.get_legend()
leg.get_texts()[0].set_text("SBM: K, best of 50")
leg.get_texts()[6].set_text("SBM: d, best of 50")
leg.get_texts()[11].set_text("RDPG: directed")
plt.xlabel("# Params (GMM params for SBMs)")
plt.ylabel("MSE")
plt.title(f"Drosophila old MB left, directed ({experiment}:{run})")

#####

plt.figure(figsize=(22, 12))

cmap = sns.light_palette("purple", as_cmap=True)
sns.scatterplot(
    data=best_sbm_df,
    x="n_params_gmm",
    y="score",
    hue="n_components_try",
    size="n_block_try",
    palette=cmap,
    **plt_kws,
)

cmap = sns.xkcd_palette(["grass green"])
s = sns.scatterplot(
    data=rdpg_df, x="n_params", y="score", hue="RDPG", palette=cmap, **plt_kws
)

leg = s.axes.get_legend()
leg.get_texts()[0].set_text("SBM: d, best of 50")
leg.get_texts()[5].set_text("SBM: K, best of 50")
leg.get_texts()[11].set_text("RDPG: directed")
plt.xlabel("# Params (GMM params for SBMs)")
plt.ylabel("MSE")
plt.title(f"Drosophila old MB left, directed ({experiment}:{run})")


#%%

