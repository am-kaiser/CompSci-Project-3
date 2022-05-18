import sklearn.gaussian_process as gp


def define_gpr_alg(y_train, x_train, x_test):
    # Define kernel
    kernel = 1 * gp.kernels.RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
    gaussianprocess = gp.GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9, normalize_y=True)
    # Fit Gaussian process
    gaussianprocess.fit(x_train, y_train)
    gaussianprocess.kernel_
    # Make prediction
    mean_pred, std_pred = gaussianprocess.predict(x_test, return_std=True)

    return mean_pred.flatten(), std_pred.flatten()