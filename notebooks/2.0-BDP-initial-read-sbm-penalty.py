#%%
import os
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
from IPython import get_ipython  # just to decieve flake8
import ijson
import src.utils as utils

base_path = "./maggot_models/simulations/runs/"
experiment = "sbm_rss_lik_sim"
run = 1
sbm_df = utils.load_config(base_path, experiment, run)

#%%
