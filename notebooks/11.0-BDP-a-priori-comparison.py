#%%
import numpy as np
import pandas as pd

from src.utils import get_best
from src.data import load_left, load_right
from src.models import fit_a_priori, select_dcsbm, select_rdpg, select_sbm
from graspy.plot import heatmap
from graspy.models import DCSBMEstimator, RDPGEstimator, SBMEstimator
import seaborn as sns

#%%
show_plots = False
score = "mse"

left_adj, left_labels = load_left()
if show_plots:
    heatmap(left_adj, inner_hier_labels=left_labels, cbar=False)

estimator = SBMEstimator()
ap_results = fit_a_priori(estimator, left_adj, left_labels)

param_grid = dict(n_blocks=list(range(1, 10)))
sweep_results = select_sbm(left_adj, param_grid, n_init=25, n_jobs=-2)

sweep_results

best_results = get_best(
    sweep_results, "n_params", score_name=score, small_better=False
)
#%%
sns.scatterplot(data=best_results, x="n_params", y=score)
sns.scatterplot(data=ap_results, x="n_params", y=score)

#%%
estimator = DCSBMEstimator()
ap_results = fit_a_priori(estimator, left_adj, left_labels)

param_grid = dict(n_blocks=list(range(1, 10)))
sweep_results = select_dcsbm(left_adj, param_grid, n_init=25, n_jobs=-2, metric=None)

sweep_results

best_results = get_best(
    sweep_results, "n_params", score_name=score, small_better=False
)
#%%
sns.scatterplot(data=best_results, x="n_params", y=score)
sns.scatterplot(data=ap_results, x="n_params", y=score)

#%%

#%%
dc = DCSBMEstimator()
dc.fit(left_adj, y=left_labels)
dc._n_parameters()
dc.block_p_

left_adj.shape

#%%
209 + 

#%%
