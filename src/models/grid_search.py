from sklearn.model_selection import GridSearchCV
import numpy as np
import pandas as pd
from tqdm import tqdm


def format_columns(string):
    if isinstance(string, str):
        # if "mean_test_" in string:
        #     string = string.replace("mean_test_", "")
        if "mean_" in string:
            string = string.replace("mean_", "")
        if "test_" in string:
            string = string.replace("test_", "")
        # elif "mean_" in string:
        #     string = string.replace("mean_", "")
    return string


def remove_columns(df):
    # TODO implement
    # columns
    columns = df.columns.values
    remove = []
    bad_keys = ["std_", "split0_"]
    for column in columns:
        for bad_key in bad_keys:
            if bad_key in column:
                remove.append(column)
                continue
    df.drop(columns=remove, inplace=True)
    return df


class DummyCV:
    def __init__(self, n_init=1):
        self.n_splits = n_init

    def split(self, X, y, groups=None):
        for i in range(self.n_splits):
            yield (np.arange(X.shape[0]), np.arange(X.shape[0]))

    def get_n_splits(self, X, y, groups=None):
        return self.n_splits


class GridSearchUS(GridSearchCV):
    def __init__(
        self,
        estimator,
        param_grid,
        scoring=None,
        n_jobs=None,
        refit=False,
        verbose=0,
        pre_dispatch="2*n_jobs",
        error_score="raise-deprecating",
        return_train_score=False,
        n_init=6,
    ):
        super().__init__(
            estimator,
            param_grid,
            scoring=scoring,
            n_jobs=n_jobs,
            iid=False,
            refit=refit,
            cv=DummyCV(),
            verbose=verbose,
            pre_dispatch=pre_dispatch,
            error_score=error_score,
            return_train_score=False,
        )
        self.n_init = n_init

    def fit(self, X, y=None):
        results = []
        for i in tqdm(range(self.n_init)):
            sup = super().fit(X, y=y)
            temp_results = pd.DataFrame.from_dict(sup.cv_results_)
            temp_results = remove_columns(temp_results)
            temp_results.rename(columns=format_columns, inplace=True)
            results.append(temp_results)
        results_df = pd.concat(results, ignore_index=True)
        self.cv_results_ = results_df
        return self
