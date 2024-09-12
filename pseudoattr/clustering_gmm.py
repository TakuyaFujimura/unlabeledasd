import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import GridSearchCV


def gmm_aic_score(estimator, X):
    """Callable to pass to GridSearchCV that will use the AIC score."""
    # Make it negative since GridSearchCV expects a score to maximize
    return -estimator.aic(X)


def gmm_bic_score(estimator, X):
    """Callable to pass to GridSearchCV that will use the BIC score."""
    # Make it negative since GridSearchCV expects a score to maximize
    return -estimator.bic(X)


def get_info_df(grid_search):
    df = pd.DataFrame(grid_search.cv_results_)[
        ["param_n_components", "param_covariance_type", "mean_test_score"]
    ]
    df = df.sort_values(by="mean_test_score")
    return df


def gmm_clustering(Z, n_cluster, ic):
    param_grid = {
        "n_components": range(1, n_cluster + 1),
        "covariance_type": ["full"],
    }
    if ic == "bic":
        grid_search = GridSearchCV(
            GaussianMixture(), param_grid=param_grid, scoring=gmm_bic_score
        )
    elif ic == "aic":
        grid_search = GridSearchCV(
            GaussianMixture(), param_grid=param_grid, scoring=gmm_aic_score
        )
    else:
        raise ValueError("ic must be 'bic' or 'aic'")
    grid_search.fit(Z)
    idx_list = grid_search.predict(Z)
    info_df = get_info_df(grid_search)
    return idx_list, info_df
