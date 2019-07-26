#%%
import numpy as np
import pandas as pd

from src.data import load_left, load_right
from src.models import fit_a_priori, select_dcsbm, select_rdpg, select_sbm
from graspy.plot import heatmap
from graspy.models import DCSBMEstimator, RDPGEstimator, SBMEstimator

#%%
show_plots = False


left_adj, left_labels = load_left()
if show_plots:
    heatmap(left_adj, inner_hier_labels=left_labels, cbar=False)

estimator = SBMEstimator()
ap_results = fit_a_priori(estimator, left_adj, left_labels)

param_grid = dict(n_blocks=list(range(1, 10)))
sweep_results = select_sbm(left_adj, param_grid, n_init=25, n_jobs=-2)


#%%
