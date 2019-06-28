#%%
from graspy.models import SBMEstimator
from src.data import load_new_left
from graspy.plot import heatmap
import numpy as np

adj, labels = load_new_left()

sbm = SBMEstimator(loops=False, co_block=False)
sbm.fit(adj, y=labels)
heatmap(sbm.p_mat_, inner_hier_labels=labels, vmin=0, vmax=1)

#%%
co_labels = np.stack((labels, labels), axis=1).astype("U3")

for i, row in enumerate(co_labels):
    if row[1] == "O" or row[1] == "I":
        co_labels[i, 1] = "O/I"
co_labels

#%%
cosbm = SBMEstimator(loops=False, co_block=True)
cosbm.fit(adj, y=co_labels)
heatmap(cosbm.p_mat_, inner_hier_labels=labels)

#%%
