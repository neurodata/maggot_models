from sklearn.model_selection import GridSearchCV
import numpy as np
import pandas as pd


def format_columns(string):
    if isinstance(string, str):
        if "mean_test_" in string:
            string = string.replace("mean_test_", "")
    return string


def remove_columns(df):
    # TODO implement
    # columns
    # out_df.drop(
    #     columns=[
    #         "std_fit_time",
    #         "std_score_time",
    #         "split0_test_score",
    #         "std_test_score",
    #     ],
    #     inplace=True,
    # )
    return df


class DummyCV:
    def __init__(self):
        self.n_splits = 1

    def split(self, X, y, groups=None):
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

    def fit(self, X, y=None):
        sup = super().fit(X, y=y)
        cv_results = pd.DataFrame.from_dict(sup.cv_results_)
        cv_results = remove_columns(cv_results)
        cv_results.rename(columns=format_columns, inplace=True)
        self.cv_results_ = cv_results
        return self
