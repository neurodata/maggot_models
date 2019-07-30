from os.path import basename
from pathlib import Path

import numpy as np
import pandas as pd
from sacred import Experiment
from sacred.observers import FileStorageObserver, SlackObserver
from graspy.models import SBMEstimator, DCSBMEstimator
from src.data import load_left, load_right
from graspy.utils import binarize, symmetrize, cartprod
from src.models import fit_a_priori
from src.utils import save_obj

ex = Experiment("Fit a priori to opposite hemisphere")

current_file = basename(__file__)[:-3]

sacred_file_path = Path(f"./maggot_models/models/runs/{current_file}")

slack_obs = SlackObserver.from_config("slack.json")

file_obs = FileStorageObserver.create(sacred_file_path)

ex.observers.append(slack_obs)
ex.observers.append(file_obs)


@ex.config
def config():
    # Variables defined in config get automatically passed to main
    pass


def get_block_indices(y):
    """
    y is a length n_verts vector of labels

    returns a length n_verts vector in the same order as the input
    indicates which block each node is
    """
    block_labels, block_inv, block_sizes = np.unique(
        y, return_inverse=True, return_counts=True
    )

    n_blocks = len(block_labels)
    block_inds = range(n_blocks)

    block_vert_inds = []
    for i in block_inds:
        # get the inds from the original graph
        inds = np.where(block_inv == i)[0]
        block_vert_inds.append(inds)
    return block_vert_inds, block_inds, block_inv


def block_to_full(block_mat, inverse, shape):
    """
    "blows up" a k x k matrix, where k is the number of communities, 
    into a full n x n probability matrix

    block mat : k x k 
    inverse : array like length n, 
    """
    block_map = cartprod(inverse, inverse).T
    mat_by_edge = block_mat[block_map[0], block_map[1]]
    full_mat = mat_by_edge.reshape(shape)
    return full_mat


def mse_on_other(estimator, graph, labels):
    block_p = estimator.block_p_

    block_vert_inds, block_inds, block_inv = get_block_indices(labels)
    pred_p_mat = block_to_full(block_p, block_inv, graph.shape)
    estimator.p_mat_ = pred_p_mat
    mse = estimator.mse(graph)
    return mse


def likelihood_on_other(estimator, graph, labels, clip=0):
    block_p = estimator.block_p_

    block_vert_inds, block_inds, block_inv = get_block_indices(labels)
    pred_p_mat = block_to_full(block_p, block_inv, graph.shape)
    estimator.p_mat_ = pred_p_mat
    mse = estimator.score(graph, clip=clip)
    return mse


def run_fit(seed):
    np.random.seed(seed)

    # load
    left_graph, left_labels = load_left()
    right_graph, right_labels = load_right()

    # fit SBM left, predict right
    sbm_fit_left = SBMEstimator(directed=True, loops=False)
    sbm_fit_left.fit(left_graph, y=left_labels)
    right_pred_mse = mse_on_other(sbm_fit_left, right_graph, right_labels)
    right_pred_likelihood = likelihood_on_other(sbm_fit_left, right_graph, right_labels)
    right_pred_sc_likelihood = likelihood_on_other(
        sbm_fit_left,
        right_graph,
        right_labels,
        clip=1 / (right_graph.size - right_graph.shape[0]),
    )
    right_pred_dict = {
        "n_params": sbm_fit_left._n_parameters(),
        "mse": right_pred_mse,
        "likelihood": right_pred_likelihood,
        "zc_likelihood": right_pred_likelihood,
        "sc_likelihood": right_pred_sc_likelihood,
    }
    right_pred_df = pd.DataFrame(right_pred_dict, index=[0])
    print(right_pred_df)
    save_obj(right_pred_df, file_obs, "right_pred_sbm_df")

    # fit SBM right, predict left
    sbm_fit_right = SBMEstimator(directed=True, loops=False)
    sbm_fit_right.fit(right_graph, y=right_labels)
    left_pred_mse = mse_on_other(sbm_fit_right, left_graph, left_labels)
    left_pred_likelihood = likelihood_on_other(sbm_fit_right, left_graph, left_labels)
    left_pred_sc_likelihood = likelihood_on_other(
        sbm_fit_right,
        left_graph,
        left_labels,
        clip=1 / (left_graph.size - left_graph.shape[0]),
    )
    left_pred_dict = {
        "n_params": sbm_fit_right._n_parameters(),
        "mse": left_pred_mse,
        "likelihood": left_pred_likelihood,
        "zc_likelihood": left_pred_likelihood,
        "sc_likelihood": left_pred_sc_likelihood,
    }
    left_pred_df = pd.DataFrame(left_pred_dict, index=[0])
    print(left_pred_df)
    save_obj(left_pred_df, file_obs, "left_pred_sbm_df")
    # sbm_fit_right = SBMEstimator(directed=True, loops=False)
    # sbm_fit_right.fit(right_graph, y=right_labels)
    # right_b = sbm_fit_right.block_p_

    # # save_obj(sbm_left_df, file_obs, "sbm_left_df")

    return 0


@ex.automain
def main(seed):
    seed = 8888
    out = run_fit(seed)
    return out
