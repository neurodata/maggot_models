import numpy as np
import pandas as pd
from graspologic.models import DCSBMEstimator, SBMEstimator
from graspologic.utils import binarize, remove_loops
from scipy.stats import poisson


def calculate_blockmodel_likelihood(
    adj, labels, lp_inds, rp_inds, pad_probabilities=True, models=["DCSBM"]
):
    rows = []
    left_adj = binarize(adj[np.ix_(lp_inds, lp_inds)])
    left_adj = remove_loops(left_adj)
    right_adj = binarize(adj[np.ix_(rp_inds, rp_inds)])
    right_adj = remove_loops(right_adj)
    for model, name in zip([DCSBMEstimator], ["DCSBM"]):
        estimator = model(directed=True, loops=False)
        uni_labels, inv = np.unique(labels, return_inverse=True)
        estimator.fit(left_adj, inv[lp_inds])
        train_left_p = estimator.p_mat_
        if pad_probabilities:
            train_left_p[train_left_p == 0] = 1 / train_left_p.size

        n_params = estimator._n_parameters() + len(labels)

        score = poisson.logpmf(left_adj, train_left_p).sum()
        rows.append(
            dict(
                train_side="Left",
                test="Same",
                test_side="Left",
                score=score,
                model=name,
                n_params=n_params,
                norm_score=score / left_adj.sum(),
                n_communities=len(uni_labels),
            )
        )
        score = poisson.logpmf(right_adj, train_left_p).sum()
        rows.append(
            dict(
                train_side="Left",
                test="Opposite",
                test_side="Right",
                score=score,
                model=name,
                n_params=n_params,
                norm_score=score / right_adj.sum(),
                n_communities=len(uni_labels),
            )
        )

        estimator = model(directed=True, loops=False)
        estimator.fit(right_adj, inv[rp_inds])
        train_right_p = estimator.p_mat_
        if pad_probabilities:
            train_right_p[train_right_p == 0] = 1 / train_right_p.size

        n_params = estimator._n_parameters() + len(labels)

        score = poisson.logpmf(left_adj, train_right_p).sum()
        rows.append(
            dict(
                train_side="Right",
                test="Opposite",
                test_side="Left",
                score=score,
                model=name,
                n_params=n_params,
                norm_score=score / left_adj.sum(),
                n_communities=len(uni_labels),
            )
        )
        score = poisson.logpmf(right_adj, train_right_p).sum()
        rows.append(
            dict(
                train_side="Right",
                test="Same",
                test_side="Right",
                score=score,
                model=name,
                n_params=n_params,
                norm_score=score / right_adj.sum(),
                n_communities=len(uni_labels),
            )
        )
    return pd.DataFrame(rows)
