from itertools import product

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator
from sklearn.model_selection import ParameterGrid


class GridSweep(BaseEstimator):
    def __init__(
        self,
        estimator,
        param_grid,
        scoring=None,
        n_jobs=1,
        refit=False,
        verbose=0,
        pre_dispatch="2*n_jobs",
        n_init=1,
        seed=None,
        small_better=True,
    ):
        # TODO input validation
        self.estimator = estimator
        self.param_grid = param_grid
        self.scoring = scoring
        self.n_jobs = n_jobs
        self.refit = refit
        self.verbose = verbose
        self.pre_dispatch = pre_dispatch
        self.n_init = n_init
        self.seed = seed
        self.small_better = small_better

    def fit(self, X, y=None):
        np.random.seed(self.seed)

        if not isinstance(self.param_grid, list):
            param_grid = list(ParameterGrid(self.param_grid))

        estimator = self.estimator

        seeds = np.random.randint(1e8, size=self.n_init)

        def estimator_fit(params, seed):
            np.random.seed(seed)
            model = estimator(**params)
            model.fit(X, y=y)
            return model

        parallel = Parallel(
            n_jobs=self.n_jobs, verbose=self.verbose, pre_dispatch=self.pre_dispatch
        )

        if self.verbose > 0:
            print("Fitting models...")
        models = parallel(
            delayed(estimator_fit)(p, s) for p, s in product(param_grid, seeds)
        )

        scorers = self.scoring

        def estimator_score(model):
            scores = {}
            for name, scorer in scorers.items():
                scores[name] = scorer(model, X, y=y)
            return scores

        if self.verbose > 0:
            print("Scoring models...")

        model_scores = parallel(delayed(estimator_score)(m) for m in models)

        result_df = pd.DataFrame(model_scores)
        self.result_df_ = result_df

        if self.refit is not False:
            if self.small_better:
                best_ind = result_df[self.refit].idxmin()
            else:
                best_ind = result_df[self.refit].idxmax()

            self.model_ = models[best_ind]

        return self
