from graspy.inference import LatentDistributionTest

from os.path import basename
from pathlib import Path

import numpy as np
from sacred import Experiment
from sacred.observers import FileStorageObserver, SlackObserver
from graspy.models import SBMEstimator, DCSBMEstimator
from src.data import load_left, load_right
from graspy.utils import binarize, symmetrize
from src.models import fit_a_priori
from src.utils import save_obj
import pandas as pd

from joblib import Parallel, delayed

ex = Experiment("Run LDT")

current_file = basename(__file__)[:-3]

sacred_file_path = Path(f"./maggot_models/models/runs/{current_file}")

slack_obs = SlackObserver.from_config("slack.json")

file_obs = FileStorageObserver.create(sacred_file_path)

ex.observers.append(slack_obs)
ex.observers.append(file_obs)


@ex.config
def config():
    # Variables defined in config get automatically passed to main
    n_components_range = list(range(1, 55))  # noqa
    directed = True  # noqa: F841


def fit_ldt(left_graph, right_graph, n_components, n_bootstraps=500):
    ldt = LatentDistributionTest(n_components=n_components, n_bootstraps=n_bootstraps)
    ldt.fit(left_graph, right_graph)
    result = {}
    result["p-value"] = ldt.p_
    result["sample-t"] = ldt.sample_T_statistic_
    result["n_components"] = n_components
    print(f"Done with {n_components}")
    return result


def run_fit(seed, directed, n_components_range):
    # run left
    left_graph, labels = load_left()
    if not directed:
        left_graph = symmetrize(left_graph, method="avg")

    # run right
    right_graph, labels = load_right()
    if not directed:
        right_graph = symmetrize(right_graph, method="avg")

    def fit(n_components):
        # np.random.seed(seed)
        return fit_ldt(left_graph, right_graph, n_components)

    outs = Parallel(n_jobs=-2, verbose=5)(delayed(fit)(n) for n in n_components_range)

    out_df = pd.DataFrame(outs)
    save_obj(out_df, file_obs, "ldt_df")
    return 0


@ex.automain
def main(seed, directed, n_components_range):
    seed = 8888

    out = run_fit(seed, directed, n_components_range)
    return out
