#%%
from os.path import basename
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sacred import Experiment
from sacred.observers import FileStorageObserver, SlackObserver

# from src.utils import gen_B, gen_sbm, select_sbm
import json

sacred_file_path = Path(f"./maggot_models/simulations/runs/sbm_rss_lik_sim/2/run.json")

f = open(str(sacred_file_path), mode="r")
out = json.load(f)
f.close()
data_dict = out["result"]["values"]
df = pd.DataFrame.from_dict(data_dict)
df
#%%
