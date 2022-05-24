import sklearn.gaussian_process as gp
from sklearn.model_selection import GridSearchCV
import numpy as np


def search_best_model(obs_train, grid_train, full_grid, get_std=False):
    # Define grid for gridsearch
    param_grid = [
        {"kernel": [gp.kernels.RBF(length_scale=ls, length_scale_bounds=(1e-5, 1e5)) for ls in np.linspace(-5, 5, 11)]},
        {"kernel": [gp.kernels.ExpSineSquared(length_scale=ls, length_scale_bounds=(1e-5, 1e5)) for ls in
                    np.logspace(-1, 1, 2)]}]

    # Define regressor
    gaussian_process = gp.GaussianProcessRegressor(n_restarts_optimizer=10)

    # Perform grid search
    clf = GridSearchCV(estimator=gaussian_process, param_grid=param_grid, scoring='r2')

    # Fit Gaussian process
    clf.fit(grid_train, obs_train)

    # Print best model and scores
    print('Best model with parameters: ' + str(clf.best_params_) + ' and R2: ' + str(clf.best_score_))

    # Make prediction
    if get_std:
        mean_pred, cov_std_pred = gaussian_process.predict(full_grid, return_std=True)
    else:
        mean_pred, cov_std_pred = gaussian_process.predict(full_grid, return_cov=True)

    return mean_pred, cov_std_pred


def perform_gpr_alg(obs_train, grid_train, full_grid,
                    kernel=gp.kernels.RBF(length_scale=1.0, length_scale_bounds=(1e-5, 1e5)), get_std=False):
    # Define regressor
    gaussianprocess = gp.GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)
    # Fit Gaussian process
    gaussianprocess.fit(grid_train, obs_train)
    print("Model Parameters: ", gaussianprocess.kernel_.get_params())

    # Make prediction
    if get_std:
        mean_pred, cov_std_pred = gaussianprocess.predict(full_grid, return_std=True)
    else:
        mean_pred, cov_std_pred = gaussianprocess.predict(full_grid, return_cov=True)

    return mean_pred, cov_std_pred
