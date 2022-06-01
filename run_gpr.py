import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.gaussian_process as gp
from sklearn.model_selection import train_test_split

from gpr_alg import prepare_data, plot_data, perform_gpr
import pickle

def perform_1D_gpr(make_grid_search=True, kernel=gp.kernels.RBF(length_scale=1.0, length_scale_bounds=(1e-5, 1e5))):
    """Function to perform gaussian process regression (gpr) for one 1D input data
    :param make_grid_search: perform grid search True or False
    :param kernel: kernel for gpr"""
    # Make grid
    grid = prepare_data.create_1D_grid(step_size=0.01)
    # Create data
    data = prepare_data.create_data(func_param=[1, grid], func_name='zhouetal')
    # Scale data (i.e. normalize)
    data_scaled, _ = prepare_data.rescale_data(data, type='standardization')
    # Split data into test and training datasets
    grid_train, grid_test, data_train, data_test = prepare_data.split_train_test_data(grid, data_scaled, train_perc=0.3)

    # Search for best model with a grid search
    if make_grid_search:
        list_kernels = []
        list_stats = []
        list_mean_pred = []
        list_cov_pred = []

        for index in range(1, 21, 1):
            temp_kernels, temp_stats, temp_mean_pred, temp_cov_pred = perform_gpr.search_best_model(
                obs_train=data_train, grid_train=grid_train,
                full_grid=grid,
                full_data=data_scaled, get_std=True)

            list_kernels.extend(temp_kernels)
            list_stats.extend(temp_stats)
            list_mean_pred.extend(temp_mean_pred)
            list_cov_pred.extend(temp_cov_pred)

            print('Iteration: ' + str(index))

        # Save statistics in dataframe
        stats_df = pd.DataFrame(list(zip(list_kernels, list_stats, list_mean_pred, list_cov_pred)),
                                columns=['kernel', 'stats', 'pred_mean', 'pred_cov'])

        # Remove duplicate rows
        stats_df = stats_df[~stats_df['pred_mean'].apply(tuple).duplicated()]
        stats_df.to_pickle('grid_search_stats_1D_100_iterations.pkl')

    # Perform Gaussian process regression for specific kernel
    else:
        # Plot original data
        plot_data.make_2D_plot(data, grid, y_label='f(x)', file_name='data_zhouetal.png')

        # Fit GPR
        mean_prediction, std_prediction = perform_gpr.perform_gpr_alg(obs_train=data_train, grid_train=grid_train,
                                                                      full_grid=grid, get_std=True, kernel=kernel)

        # Plot fitted data plus confidence interval
        plot_data.make_2D_conf_interval_plot(grid_values=grid, obs_values=data_scaled, grid_values_train=grid_train,
                                             obs_values_train=data_train, mean_pred=mean_prediction.flatten(),
                                             std_pred=std_prediction.flatten())
        plt.show()


def create_2D_data(add_noise=False, apply_conditions=True):
    """Function to create grid and data and split both for training
    :param add_noise: add noise to data True or False
    :param apply_conditions: apply initial and boundary conditions to train dataset True or False
    """
    # Make grid
    # Note: x1=x, x2=t
    grid_x1, grid_x2 = prepare_data.create_2D_grid(x1_step_size=0.05, x2_step_size=0.05)

    # Reshape grid values
    grid_resh = np.stack([grid_x1.flatten(), grid_x2.flatten()])

    # Create data
    data = prepare_data.create_data(func_param=[grid_x1, grid_x2], func_name='heat_equation', add_noise=add_noise)

    #Noiseless data used for initial conditions
    data_noiseless = prepare_data.create_data(func_param=[grid_x1, grid_x2],
                                              func_name='heat_equation', add_noise=False)
    # Split data into test and training datasets
    # Include initial and boundary conditions in train dataset
    if apply_conditions:
        print('applying conditions')
        # Initial values
        grid_ini_x1 = grid_x1[:1, :]
        grid_ini_x2 = grid_x2[:1, :]
        grid_ini = np.array([grid_ini_x1.flatten(), grid_ini_x2.flatten()]).T

        data_ini = data_noiseless[:1, :].reshape(-1, 1)

        # Lower boundary conditions
        grid_lbound_x1 = grid_x1[:, :1]
        grid_lbound_x2 = grid_x2[:, :1]
        grid_lbound = np.array([grid_lbound_x1.flatten(), grid_lbound_x2.flatten()]).T

        data_lbound = data_noiseless[:, :1].reshape(-1, 1)

        # Upper boundary conditions
        grid_hbound_x1 = grid_x1[:, -1:]
        grid_hbound_x2 = grid_x2[:, -1:]
        grid_hbound = np.array([grid_hbound_x1.flatten(), grid_hbound_x2.flatten()]).T

        data_hbound = data_noiseless[:, -1:].reshape(-1, 1)

        # Split data in train dataset (without initial and boundary conditions)
        grid_train, _, data_train, _ = train_test_split(
            np.hstack([grid_x1[1:, 1:-1].reshape(-1, 1), grid_x2[1:, 1:-1].reshape(-1, 1)]),
            data[1:, 1:-1].reshape(-1, 1), test_size=0.25, random_state=2022)

        # Add initial and boundary conditions to train dataset
        grid_train = np.concatenate([grid_train, grid_ini, grid_lbound, grid_hbound])
        data_train = np.concatenate([data_train, data_ini, data_lbound, data_hbound])

        print('train set size: ' + str(np.shape(grid_train)[0] / np.shape(grid_x1.flatten())[0]))

    else:
        grid_train, _, data_train, _ = train_test_split(np.hstack([grid_x1.reshape(-1, 1), grid_x2.reshape(-1, 1)]),
                                                        data.reshape(-1, 1), test_size=0.2, random_state=2022)

    # Scale data (i.e. normalize)
    data_scaled, scaler = prepare_data.rescale_data(data.reshape(-1, 1), type='standardization')
    data_train_scaled = scaler.transform(data_train)

    return grid_x1, grid_x2, grid_train, grid_resh, data_train_scaled, data_scaled


def perform_2D_gpr(make_grid_search=True, kernel=gp.kernels.RBF(length_scale=1.0, length_scale_bounds=(1e-5, 1e5)),
                   add_noise=False):
    """Function to perform gaussian process regression (gpr) for one 2D input data
    :param make_grid_search: perform grid search True or False
    :param kernel: kernel for gpr
    :param add_noise: add noise to data True or False"""
    # Load grid and data
    grid_x1, grid_x2, grid_train, full_grid, data_train, data = create_2D_data(add_noise)
    # Search for best model with a grid search
    if make_grid_search:
        list_kernels = []
        list_stats = []
        list_mean_pred = []
        list_cov_pred = []

        for index in range(1, 101, 1):
            temp_kernels, temp_stats, temp_mean_pred, temp_cov_pred = perform_gpr.search_best_model(
                obs_train=data_train, grid_train=grid_train,
                full_grid=full_grid.T,
                full_data=data)

            list_kernels.extend(temp_kernels)
            list_stats.extend(temp_stats)
            list_mean_pred.extend(temp_mean_pred)
            list_cov_pred.extend(temp_cov_pred)

            print('Iteration: ' + str(index))

        # Save statistics in dataframe
        stats_df = pd.DataFrame(list(zip(list_kernels, list_stats, list_mean_pred, list_cov_pred)),
                                columns=['kernel', 'stats', 'pred_mean', 'pred_cov'])

        # Remove duplicate rows
        stats_df = stats_df[~stats_df['pred_mean'].apply(tuple).duplicated()]

        # Save dataframe
        stats_df.to_pickle('grid_search_stats_2D_100_iterations_with_noise.pkl')

    # Perform Gaussian process regression for specific kernel
    else:
        # Plot original  data
        plot_data.make_3D_surface_plot(x=grid_x1, y=grid_x2,
                                       z=data.reshape(np.shape(grid_x1)[0], np.shape(grid_x1)[1]))

        # Perform GPR
        mean_prediction, cov_prediction, _ = perform_gpr.perform_gpr_alg(obs_train=data_train,
                                                                         grid_train=grid_train,
                                                                         full_grid=full_grid.T,
                                                                         kernel=kernel)
        # Plot prediction
        plot_data.make_3D_surface_plot(x=grid_x1, y=grid_x2,
                                       z=mean_prediction.reshape(np.shape(grid_x1)[0], np.shape(grid_x1)[1]))

        # Plot posteriors
        posteriors = plot_data.plot_posteriors(x=grid_x1, y=grid_x2,
                                               z=data.reshape(np.shape(grid_x1)[0], np.shape(grid_x1)[1]),
                                               mean_pred=mean_prediction.flatten(),
                                               cov_pred=cov_prediction,
                                               posterior_nums=5, add_train_ind=False, x_train_val=grid_train)

        # plot_data.plot_posteriors_squared_error(x=grid_x1, y=grid_x2, z=data, posteriors=posteriors)

        plt.show()


if __name__ == "__main__":
    import warnings

    warnings.filterwarnings('ignore')

    tic = time.perf_counter()
    # perform_1D_gpr(make_grid_search=True)
    perform_2D_gpr(make_grid_search=True, add_noise=True)
    toc = time.perf_counter()
    print(f"Run time: {toc - tic:0.4f} seconds")
