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

ex = Experiment("Fit a priori")

current_file = basename(__file__)[:-3]

sacred_file_path = Path(f"./maggot_models/models/runs/{current_file}")

slack_obs = SlackObserver.from_config("slack.json")

file_obs = FileStorageObserver.create(sacred_file_path)

ex.observers.append(slack_obs)
ex.observers.append(file_obs)


@ex.config
def config():
    # Variables defined in config get automatically passed to main

    directed = True  # noqa: F841


def run_fit(seed, directed):
    # run left
    graph, labels = load_left()
    if not directed:
        graph = symmetrize(graph, method="avg")

    # fit SBM
    sbm = SBMEstimator(directed=True, loops=False)
    sbm_left_df = fit_a_priori(sbm, graph, labels)
    save_obj(sbm_left_df, file_obs, "sbm_left_df")

    # fit DCSBM
    dcsbm = DCSBMEstimator(directed=True, loops=False, degree_directed=False)
    dcsbm_left_df = fit_a_priori(dcsbm, graph, labels)
    save_obj(dcsbm_left_df, file_obs, "dcsbm_left_df")

    # fit dDCSBM
    ddcsbm = DCSBMEstimator(directed=True, loops=False, degree_directed=True)
    ddcsbm_left_df = fit_a_priori(ddcsbm, graph, labels)
    save_obj(ddcsbm_left_df, file_obs, "ddcsbm_left_df")

    # run right
    graph, labels = load_right()
    if not directed:
        graph = symmetrize(graph, method="avg")

    # fit SBM
    sbm = SBMEstimator(directed=True, loops=False)
    sbm_right_df = fit_a_priori(sbm, graph, labels)
    save_obj(sbm_right_df, file_obs, "sbm_right_df")

    # fit DCSBM
    dcsbm = DCSBMEstimator(directed=True, loops=False, degree_directed=False)
    dcsbm_right_df = fit_a_priori(dcsbm, graph, labels)
    save_obj(dcsbm_right_df, file_obs, "dcsbm_right_df")

    # fit dDCSBM
    ddcsbm = DCSBMEstimator(directed=True, loops=False, degree_directed=True)
    ddcsbm_right_df = fit_a_priori(ddcsbm, graph, labels)
    save_obj(ddcsbm_right_df, file_obs, "ddcsbm_right_df")

    return 0


@ex.automain
def main(seed, directed):
    seed = 8888
    out = run_fit(seed, directed)
    return out
