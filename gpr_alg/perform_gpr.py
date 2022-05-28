import re

import numpy as np
import sklearn.gaussian_process as gp
from sklearn.model_selection import GridSearchCV


def search_best_model_scikit(obs_train, grid_train, full_grid, get_std=False):
    # Define grid for gridsearch
    param_grid = [
        {"kernel": [gp.kernels.RBF(length_scale=ls, length_scale_bounds=(1e-5, 1e5)) for ls in np.linspace(0, 1, 11)]},
        {"kernel": [gp.kernels.RationalQuadratic(length_scale=ls, length_scale_bounds=(1e-5, 1e5)) for ls in
                    np.linspace(0, 1, 11)]},
        {"kernel": [gp.kernels.ExpSineSquared(length_scale=ls, length_scale_bounds=(1e-5, 1e5)) for ls in
                    np.linspace(0, 1, 11)]},
        {"kernel": [gp.kernels.Sum((gp.kernels.ExpSineSquared(length_scale=ls, length_scale_bounds=(1e-5, 1e5)) for ls in
                    np.linspace(0, 1, 11)), (gp.kernels.WhiteKernel(noise_level=nl) for nl in np.linspace(0, 1, 11)))]}]

    # Define regressor
    # Note: a value (alpha) larger than machine epsilon is added to the diagonal to ensure positive semi definite
    gaussian_process = gp.GaussianProcessRegressor(n_restarts_optimizer=10, alpha=3e-7)

    # Perform grid search
    clf = GridSearchCV(estimator=gaussian_process, param_grid=param_grid, scoring='neg_mean_absolute_error')

    # Fit Gaussian process
    clf.fit(grid_train, obs_train)

    # Print best model and scores
    print('Best model with parameters: ' + str(clf.best_params_) + ' and R2: '
          + str(gaussian_process.score(grid_train, obs_train)))

    # Make prediction
    if get_std:
        mean_pred, cov_std_pred = gaussian_process.predict(full_grid, return_std=True)
    else:
        mean_pred, cov_std_pred = gaussian_process.predict(full_grid, return_cov=True)

    return mean_pred, cov_std_pred


def search_best_model(obs_train, grid_train, full_grid, full_data, get_std=False):
    # Define grid for gridsearch
    param_grid = [
        {"kernel": [gp.kernels.RBF(length_scale=ls, length_scale_bounds=(1e-5, 1e5)) for ls in np.linspace(0, 5, 6)]},
        {"kernel": [gp.kernels.RationalQuadratic(length_scale=ls, length_scale_bounds=(1e-5, 1e5)) for ls in
                    np.linspace(0, 5, 6)]},
        {"kernel": [gp.kernels.ExpSineSquared(length_scale=ls, length_scale_bounds=(1e-5, 1e5)) for ls in
                    np.linspace(0, 5, 6)]},
        {"kernel": [gp.kernels.Sum(gp.kernels.RBF(length_scale=1, length_scale_bounds=(1e-5, 1e5)), gp.kernels.WhiteKernel(noise_level=1))]},
        {"kernel": [gp.kernels.Sum(gp.kernels.RationalQuadratic(length_scale=1, length_scale_bounds=(1e-5, 1e5)), gp.kernels.WhiteKernel(noise_level=1))]},
        {"kernel": [gp.kernels.Sum(gp.kernels.ExpSineSquared(length_scale=1, length_scale_bounds=(1e-5, 1e5)), gp.kernels.WhiteKernel(noise_level=1))]}]

    # Create array to save statistics for different kernels
    kernels = []
    stats = []
    pred_mean = []
    pred_cov_std = []

    # Loop over grid and calculate stats for every fit
    for kernel_type in param_grid:
        for _, kernel_type_values in kernel_type.items():
            for kernel in kernel_type_values:
                try:
                    mean_prediction, cov_std_prediction, model_params = perform_gpr_alg(obs_train, grid_train,
                                                                                        full_grid, kernel=kernel,
                                                                                        get_std=get_std)

                    # Store information of optimized kernel
                    type_of_kernel = re.sub(r'\([^)]*\)', '', str(kernel))

                    if 'length_scale_bounds' in model_params:
                        del model_params['length_scale_bounds']

                    if 'periodicity_bounds' in model_params:
                        del model_params['periodicity_bounds']

                    if 'alpha_bounds' in model_params:
                        del model_params['alpha_bounds']

                    kernel_info = ': '.join([str(type_of_kernel), str(model_params)])

                    kernels.append(kernel_info)
                    stats.append(np.nanmean((np.abs(full_data - mean_prediction))))
                    pred_mean.append(mean_prediction.flatten())
                    pred_cov_std.append(cov_std_prediction)

                except Exception:
                    pass

    return kernels, stats, pred_mean, pred_cov_std


def perform_gpr_alg(obs_train, grid_train, full_grid,
                    kernel=gp.kernels.RBF(length_scale=1.0, length_scale_bounds=(1e-5, 1e5)), get_std=False):
    # Define regressor
    # Note that the kernel hyperparameters are optimized during fitting
    gaussianprocess = gp.GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=0, alpha=3e-7)
    # Fit Gaussian process
    gaussianprocess.fit(grid_train, obs_train)

    # Make prediction
    if get_std:
        mean_pred, cov_std_pred = gaussianprocess.predict(full_grid, return_std=True)
    else:
        mean_pred, cov_std_pred = gaussianprocess.predict(full_grid, return_cov=True)

    return mean_pred, cov_std_pred, gaussianprocess.kernel_.get_params()
