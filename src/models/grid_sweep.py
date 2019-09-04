from itertools import product

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator
from sklearn.model_selection import ParameterGrid


class GridSweep(BaseEstimator):
    """Provides functionality for sweeping over parameters using sklearn-like estimators
    
    Parameters
    ----------
    estimator : estimator object.
        This is assumed to implement the scikit-learn estimator interface.
        Either estimator needs to provide a ``score`` function,
        or ``scoring`` must be passed.

    param_grid : dict or list of dictionaries
        Dictionary with parameters names (string) as keys and lists of
        parameter settings to try as values, or a list of such
        dictionaries, in which case the grids spanned by each dictionary
        in the list are explored. This enables searching over any sequence
        of parameter settings.
    
    scoring : dict or None, default: None
        A dict with names as keys and callables as values. The callables should have the 
        signature (``estimator``, ``X``, ``y``) and return a scalar value. If ``None``, 
        the model's ``score`` method is called

    n_jobs : int or None, optional (default=None)
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    pre_dispatch : int, or string, optional
        Controls the number of jobs that get dispatched during parallel
        execution. Reducing this number can be useful to avoid an
        explosion of memory consumption when more jobs get dispatched
        than CPUs can process. This parameter can be:

            - None, in which case all the jobs are immediately
              created and spawned. Use this for lightweight and
              fast-running jobs, to avoid delays due to on-demand
              spawning of the jobs

            - An int, giving the exact number of total jobs that are
              spawned

            - A string, giving an expression as a function of n_jobs,
              as in '2*n_jobs'
    
    refit : boolean, string, or callable, default=True
        Refit an estimator using the best found parameters on the whole
        dataset.

        For multiple metric evaluation, this needs to be a string denoting the
        scorer that would be used to find the best parameters for refitting
        the estimator at the end.

        Where there are considerations other than maximum score in
        choosing a best estimator, ``refit`` can be set to a function which
        returns the selected ``best_index_`` given ``cv_results_``.

        The refitted estimator is made available at the ``best_estimator_``
        attribute and permits using ``predict`` directly on this
        ``GridSearchCV`` instance.

        Also for multiple metric evaluation, the attributes ``best_index_``,
        ``best_score_`` and ``best_params_`` will only be available if
        ``refit`` is set and all of them will be determined w.r.t this specific
        scorer. ``best_score_`` is not returned if refit is callable.

        See ``scoring`` parameter to know more about multiple metric
        evaluation.

    verbose : integer
        Controls the verbosity: the higher, the more messages.

    n_init : int, optional (default=1)
        number of random initializations to try for each parameter set

    seed : int or RandomState
        random seed to use

    small_better : bool, optional, (default=True)
        whether small values of ``scoring`` functions are better than large values. Only
        matters when using ``refit``


    Returns
    -------
    [type]
        [description]
    """

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
            if scorers is not None:
                for name, scorer in scorers.items():
                    scores[name] = scorer(model, X, y=y)
            else:
                scores = {"score": model.score(X, y=y)}
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
