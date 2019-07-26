#%%
import numpy as np
import pandas as pd

import src.utils as utils
from src.utils import get_best
from src.data import load_left, load_right
from src.models import fit_a_priori, select_dcsbm, select_rdpg, select_sbm
from graspy.plot import heatmap
from graspy.models import DCSBMEstimator, RDPGEstimator, SBMEstimator
import seaborn as sns

# Vertex agnostic
#
#

# Load SBM
base_path = "./maggot_models/models/runs/"
experiment = "fit_sbm"
run = 1
config = utils.load_config(base_path, experiment, run)
sbm_left_df = utils.load_pickle(base_path, experiment, run, "sbm_left_df")
sbm_right_df = utils.load_pickle(base_path, experiment, run, "sbm_right_df")

# Load a priori
experiment = "fit_a_priori"
run = 3
config = utils.load_config(base_path, experiment, run)
ap_sbm_left_df = utils.load_pickle(base_path, experiment, run, "sbm_left_df")
ap_sbm_right_df = utils.load_pickle(base_path, experiment, run, "sbm_right_df")

# Load opposites
experiment = "fit_opposite_a_priori"
run = 2
config = utils.load_config(base_path, experiment, run)
right_pred_sbm_df = utils.load_pickle(base_path, experiment, run, "right_pred_sbm_df")
left_pred_sbm_df = utils.load_pickle(base_path, experiment, run, "left_pred_sbm_df")

# Vertex aware
#
#

# Load DCSBM
base_path = "./maggot_models/models/runs/"
experiment = "fit_dcsbm"
run = 9  # 8
config = utils.load_config(base_path, experiment, run)
dcsbm_left_df = utils.load_pickle(base_path, experiment, run, "dcsbm_left_df")
dcsbm_right_df = utils.load_pickle(base_path, experiment, run, "dcsbm_right_df")

# Load dDCSBM
experiment = "fit_ddcsbm"
run = 3
config = utils.load_config(base_path, experiment, run)
ddcsbm_left_df = utils.load_pickle(base_path, experiment, run, "ddcsbm_left_df")
ddcsbm_right_df = utils.load_pickle(base_path, experiment, run, "ddcsbm_right_df")

# Load RDPG
experiment = "fit_rdpg"
run = 3
config = utils.load_config(base_path, experiment, run)
rdpg_left_df = utils.load_pickle(base_path, experiment, run, "rdpg_left_df")
rdpg_right_df = utils.load_pickle(base_path, experiment, run, "rdpg_right_df")

# Load a priori
experiment = "fit_a_priori"
run = 3
config = utils.load_config(base_path, experiment, run)
ap_dcsbm_left_df = utils.load_pickle(base_path, experiment, run, "dcsbm_left_df")
ap_dcsbm_right_df = utils.load_pickle(base_path, experiment, run, "dcsbm_right_df")

ap_ddcsbm_left_df = utils.load_pickle(base_path, experiment, run, "ddcsbm_left_df")
ap_ddcsbm_right_df = utils.load_pickle(base_path, experiment, run, "ddcsbm_right_df")


score_name = "mse"
best_sbm_left_df = get_best(
    sbm_left_df, param_name="param_n_blocks", score_name=score_name
)
best_sbm_right_df = get_best(
    sbm_right_df, param_name="param_n_blocks", score_name=score_name
)
best_dcsbm_left_df = get_best(
    dcsbm_left_df, param_name="param_n_blocks", score_name=score_name
)
best_dcsbm_right_df = get_best(
    dcsbm_right_df, param_name="param_n_blocks", score_name=score_name
)
best_ddcsbm_left_df = get_best(
    ddcsbm_left_df, param_name="param_n_blocks", score_name=score_name
)
best_ddcsbm_right_df = get_best(
    ddcsbm_right_df, param_name="param_n_blocks", score_name=score_name
)
best_rdpg_left_df = get_best(
    rdpg_left_df, param_name="param_n_components", score_name=score_name
)
best_rdpg_right_df = get_best(
    rdpg_right_df, param_name="param_n_components", score_name=score_name
)


#%% #############
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

best_results = get_best(sweep_results, "n_params", score_name=score, small_better=False)
#%%
sns.scatterplot(data=best_results, x="n_params", y=score)
sns.scatterplot(data=ap_results, x="n_params", y=score)

#%%
estimator = DCSBMEstimator()
ap_results = fit_a_priori(estimator, left_adj, left_labels)

param_grid = dict(n_blocks=list(range(1, 10)))
sweep_results = select_dcsbm(left_adj, param_grid, n_init=25, n_jobs=-2, metric=None)

sweep_results

best_results = get_best(sweep_results, "n_params", score_name=score, small_better=False)
#%%
sns.scatterplot(data=best_results, x="n_params", y=score)
sns.scatterplot(data=ap_results, x="n_params", y=score)

