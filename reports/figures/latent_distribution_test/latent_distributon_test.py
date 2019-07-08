#%%
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import src.utils as utils

import numpy as np

#%%# Load SBM
base_path = "./maggot_models/models/runs/"
experiment = "run_ldt"
run = 8
config = utils.load_config(base_path, experiment, run)
ldt_df = utils.load_pickle(base_path, experiment, run, "ldt_df")

#%%
p_vals = ldt_df["p-value"].values
p_vals[p_vals == 0] = 1 / 500
ldt_df["p-value"] = p_vals
#%%
sns.set_context("talk", font_scale=1)
fig, ax = plt.subplots(1, 2, figsize=(15, 5))
sns.scatterplot(data=ldt_df, x="n_components", y="sample-t", ax=ax[0])
sns.scatterplot(data=ldt_df, x="n_components", y="p-value", ax=ax[1])
plt.yscale("log")
plt.ylim([1e-3, 1.3])
plt.axhline(0.05, c="g")


#%%


#%%
