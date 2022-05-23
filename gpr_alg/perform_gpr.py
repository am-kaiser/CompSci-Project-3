import sklearn.gaussian_process as gp


def define_gpr_alg(obs_train, grid_train, full_grid, get_std=False):
    # Define kernel
    kernel = 1 * gp.kernels.RBF(length_scale=1.0, length_scale_bounds=(1e-3, 1e3))
    # normalize_y refers to the constant mean function — either zero (False) or the training data mean (True)
    gaussianprocess = gp.GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9, normalize_y=True)
    # Fit Gaussian process
    gaussianprocess.fit(grid_train, obs_train)
    print("Model Parameters: ", gaussianprocess.kernel_.get_params())
    gaussianprocess.kernel_
    # Make prediction
    if get_std:
        mean_pred, cov_std_pred = gaussianprocess.predict(full_grid, return_std=True)
    else:
        mean_pred, cov_std_pred = gaussianprocess.predict(full_grid, return_cov=True)

    return mean_pred, cov_std_pred
