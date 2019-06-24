from os.path import basename
from pathlib import Path

import numpy as np
from sacred import Experiment
from sacred.observers import FileStorageObserver, SlackObserver

from graspy.datasets import load_drosophila_left
from graspy.utils import binarize, symmetrize
from src.models import select_rdpg
from src.utils import save_obj

ex = Experiment("Fit RDPG")

current_file = basename(__file__)[:-3]

sacred_file_path = Path(f"./maggot_models/models/runs/{current_file}")

slack_obs = SlackObserver.from_config("slack.json")

file_obs = FileStorageObserver.create(sacred_file_path)

ex.observers.append(slack_obs)
ex.observers.append(file_obs)


@ex.config
def config():
    # Variables defined in config get automatically passed to main

    # Parameter range for the models
    n_components_try_range = list(range(1, 13))
    diag_aug_try_range = np.linspace(0, 10, 10)
    plus_c_try_range = np.linspace(0, 10, 10)

    param_grid = {  # noqa: F841
        "n_components": n_components_try_range,
        "diag_aug_weight": diag_aug_try_range,
        "plus_c_weight": plus_c_try_range,
    }

    # Parameters for the experiment
    n_jobs = -2  # noqa: F841
    directed = True  # noqa: F841


def run_fit(seed, param_grid, directed, n_jobs):
    graph = load_drosophila_left()
    if not directed:
        graph = symmetrize(graph, method="avg")
    graph = binarize(graph)

    np.random.seed(seed)

    rdpg_out_df = select_rdpg(graph, param_grid, directed=directed, n_jobs=n_jobs)

    save_obj(rdpg_out_df, file_obs, "rdpg_out_df")
    return 0


@ex.automain
def main(seed, param_grid, directed, n_jobs):
    seed = 8888
    out = run_fit(seed, param_grid, directed, n_jobs)
    return out
