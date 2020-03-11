#%%
import pandas as pd
import numpy as np
from graspy.embed import ClassicalMDS
import seaborn as sns
from sklearn.metrics import pairwise_distances

data_loc = "maggot_models/data/external/17-08-26L6-allC-cl.csv"
ts_df = pd.read_csv(data_loc, index_col=None)
ts_mat = ts_df.values.T
# %% [markdown]
# #

corr_mat = pairwise_distances(ts_mat, metric="correlation")

# %% [markdown]
# #
sns.clustermap(corr_mat)

# %% [markdown]
# #
from graspy.plot import pairplot

mds = ClassicalMDS(dissimilarity="precomputed")
embed = mds.fit_transform(corr_mat)
pairplot(embed)
