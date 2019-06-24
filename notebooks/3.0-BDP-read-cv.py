#%%
import os
from pathlib import Path

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

#%%
base_path = "./maggot_models/models/runs/"
experiment = "fit_dcsbm"
run = 2
config = utils.load_config(base_path, experiment, run)
dcsbm_df = utils.load_pickle(base_path, experiment, run, "dcsbm_out_df")

# dcsbm_df = dcsbm_df.apply(pd.to_numeric)
dcsbm_df.head()
#%%
def get_best(df):
    # out_df = df[(df["param_n_components"] == 1) & (df["param_regularizer"] == 0)]
    # kept_params = ["param_n_blocks"]
    param_range = np.unique(df["param_n_blocks"].values)
    best_rows = []
    for p in param_range:
        temp_df = df[df["param_n_blocks"] == p]
        ind = temp_df["mse"].idxmin()
        best_rows.append(temp_df.loc[ind, :])
    return pd.DataFrame(best_rows)


best_dcsbm_df = get_best(dcsbm_df)
# best_ddcsbm_df = get_best(ddcsbm_df)


#%%
sns.scatterplot(data=best_dcsbm_df, y="mse", x="n_params")
# sns.scatterplot(data=best_ddcsbm_df, y="mse", x="n_params")


#%%


#%%
