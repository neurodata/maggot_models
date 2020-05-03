# %% [markdown]
# ##
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

ks = np.arange(1, 100)
n = 2310

base_df = pd.DataFrame()
base_df["K"] = ks

# def k_squared(k, *args):

funcs = [lambda k, n: k ** 2, lambda k, n: k ** 2 + n, lambda k, n: k ** 2 + 2 * n]
names = [r"$K^2$", r"$K^2 + N$", r"$K^2 + 2N$"]

dfs = []
for func, name in zip(funcs, names):
    func_df = base_df.copy()
    func_df["n_params"] = func(ks, n)
    func_df["func"] = name
    dfs.append(func_df)

plot_df = pd.concat(dfs, axis=0)

sns.lineplot(data=plot_df, x="K", y="n_params", hue="func")

# %%
