# %% [markdown]
# ##
from src.visualization import adjplot, matrixplot

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_context("talk")

n_rows = 100
n_cols = 200
data11 = np.random.normal(0, 1, size=(n_rows, n_cols))
data21 = np.random.normal(0, 2, size=(n_rows, n_cols))
data13 = np.random.normal(0, 0.5, size=(n_rows, n_cols))

data12 = np.random.normal(1, 1, size=(n_rows, n_cols))
data22 = np.random.normal(1, 2, size=(n_rows, n_cols))
data23 = np.random.normal(1, 0.5, size=(n_rows, n_cols))

data = np.block([[data11, data12, data13], [data21, data22, data23]])

# %% [markdown]
# ##
fig, ax = plt.subplots(1, 1, figsize=(10, 5))
matrixplot(data, ax=ax)


# %% [markdown]
# ## Add row and column metadata
means = np.zeros(data.shape[0])
means[n_rows:] = 1

variances = np.ones(data.shape[1])
variances[n_cols : 2 * n_cols] = 2
variances[2 * n_cols :] = 0.5

fig, ax = plt.subplots(1, 1, figsize=(10, 5))
matrixplot(data, ax=ax, row_sort_class=means, col_sort_class=variances)
ax.set_xlabel("Variance")
ax.set_ylabel("Mean")


# %%
fig, ax = plt.subplots(1, 1, figsize=(10, 5))
ax, div, top_cax, left_cax = matrixplot(
    data,
    ax=ax,
    row_sort_class=means,
    col_sort_class=variances,
    row_colors=means,
    col_colors=variances,
)
top_cax.set_xlabel("Variance")
left_cax.set_ylabel("Mean")


# %%
import os

from src.io import savefig

FNAME = os.path.basename(__file__)[:-3]
print(FNAME)


def stashfig(name, **kws):
    savefig(name, foldername=FNAME, save_on=True, **kws)


s1 = pd.Series(name=1, data=[np.nan, 8.2, 8.5, 6.8, 8.4, 8.2, 8.8, np.nan])
s2 = pd.Series(name=2, data=[np.nan, 8.3, 8.1, 9.1, 9.0, 8.7, 8.9, 8.6])
s3 = pd.Series(name=3, data=[np.nan, 8.6, 8.7, 8.7, 8.6, 8.6, 8.5, 8.5])
s4 = pd.Series(name=4, data=[np.nan, 8.4, 8.8, 8.7, 8.3, 8.6, 8.4, 8.1])
s5 = pd.Series(name=5, data=[np.nan, 8.7, 8.0, 7.1, 8.9, 8.8, 9.0, 8.5])
s6 = pd.Series(name=6, data=[8.5, 8.3, 8.5, 8.4, 8.9, 8.4, 8.8, 8.8])
s7 = pd.Series(name=7, data=[np.nan, 7.6, 7.7, 7.9, 8.8, 9.3, 9.1, np.nan])
s8 = pd.Series(name=8, data=[np.nan, 8.4, 8.2, 8.1, 8.0, 8.7, 9.4, 9.5])

df = pd.concat([s1, s2, s3, s4, s5, s6, s7, s8], axis=1)

fig, ax = plt.subplots(1, 1, figsize=(8, 8))
sns.heatmap(data=df, annot=True, ax=ax, cmap="CMRmap")
ax.set_xlabel("Season")
ax.set_ylabel("Episode")
stashfig("lk")

# %% [markdown]
# ##
fig, ax = plt.subplots(1, 1, figsize=(8, 4))
sns.distplot(df.values)
ax.set_xlabel("Rating")
stashfig("lk-dist")
