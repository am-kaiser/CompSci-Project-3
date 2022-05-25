from gpr_alg import prepare_data, plot_data, perform_gpr

import sklearn.gaussian_process as gp
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score


def perform_1D_gpr(make_grid_search=True, kernel=gp.kernels.RBF(length_scale=1.0, length_scale_bounds=(1e-5, 1e5))):
    # Make grid
    grid = prepare_data.create_1D_grid(step_size=0.01)
    # Create data
    data = prepare_data.create_data(func_param=[1, grid], func_name='zhouetal')
    # Plot data
    plot_data.make_2D_plot(data, grid, y_label='f(x)', file_name='data_zhouetal.png')
    # Scale data (i.e. normalize)
    data_scaled, _ = prepare_data.rescale_data(data, type='standardization')
    # Split data into test and training datasets
    grid_train, grid_test, data_train, data_test = prepare_data.split_train_test_data(grid, data_scaled, train_perc=0.3)

    # Search for best model with a grid search
    if make_grid_search:
        mean_prediction_grid, std_prediction_grid = perform_gpr.search_best_model(obs_train=data_train,
                                                                                  grid_train=grid_train,
                                                                                  full_grid=grid, get_std=True)
        print('R2 score:' + str(r2_score(data_scaled, mean_prediction_grid)))

        plot_data.make_2D_conf_interval_plot(grid_values=grid, obs_values=data_scaled, grid_values_train=grid_train,
                                             obs_values_train=data_train, mean_pred=mean_prediction_grid.flatten(),
                                             std_pred=std_prediction_grid.flatten())
        plt.show()
    # Perform Gaussian process regression for specific kernel
    else:
        mean_prediction, std_prediction = perform_gpr.perform_gpr_alg(obs_train=data_train, grid_train=grid_train,
                                                                      full_grid=grid, get_std=True, kernel=kernel)
        print('R2 score:' + str(r2_score(data_scaled, mean_prediction)))

        plot_data.make_2D_conf_interval_plot(grid_values=grid, obs_values=data_scaled, grid_values_train=grid_train,
                                             obs_values_train=data_train, mean_pred=mean_prediction.flatten(),
                                             std_pred=std_prediction.flatten())
        plt.show()


def perform_2D_gpr(make_grid_search=True, kernel=gp.kernels.RBF(length_scale=1.0, length_scale_bounds=(1e-5, 1e5))):
    # Make grid
    # Note: x1=x, x2=t
    grid_x1, grid_x2 = prepare_data.create_2D_grid(x1_step_size=0.05, x2_step_size=0.05)

    # Reshape grid values
    grid_resh = np.stack([grid_x1.flatten(), grid_x2.flatten()])

    # Split data into test and training datasets
    size_train_set = int(np.shape(grid_x1.flatten())[0] * 0.8)
    x_train = np.stack(
        (np.random.choice(grid_x1.flatten(), size_train_set), np.random.choice(grid_x2.flatten(), size_train_set)),
        axis=-1)

    # Create data
    data = prepare_data.create_data(func_param=[grid_x1, grid_x2], func_name='heat_equation')
    data_train = prepare_data.create_data(func_param=[x_train[:, 0], x_train[:, 1]], func_name='heat_equation')

    # Scale data (i.e. normalize)
    data_scaled, scaler = prepare_data.rescale_data(data.reshape(-1, 1), type='standardization')
    data_train_scaled = scaler.transform(data_train.reshape(-1, 1))

    # Plot data
    # plot_data.make_3D_surface_plot(x=grid_x1, y=grid_x2, z=data)
    plot_data.make_3D_surface_plot(x=grid_x1, y=grid_x2,
                                   z=data_scaled.reshape(np.shape(grid_x1)[0], np.shape(grid_x1)[1]))

    # Search for best model with a grid search
    if make_grid_search:
        mean_prediction_grid, cov_prediction_grid = perform_gpr.search_best_model(obs_train=data_train_scaled,
                                                                                  grid_train=x_train,
                                                                                  full_grid=grid_resh.T)
        print('R2 score:' + str(r2_score(data_scaled.reshape(np.shape(grid_x1)[0], np.shape(grid_x1)[1]),
                                         mean_prediction_grid.reshape(np.shape(grid_x1)[0], np.shape(grid_x1)[1]))))
        # Plot prediction
        plot_data.make_3D_surface_plot(x=grid_x1, y=grid_x2,
                                       z=mean_prediction_grid.reshape(np.shape(grid_x1)[0], np.shape(grid_x1)[1]))
        # Plot posteriors
        posteriors = plot_data.plot_posteriors(x=grid_x1, y=grid_x2, z=data, mean_pred=mean_prediction_grid.flatten(),
                                               cov_pred=cov_prediction_grid,
                                               posterior_nums=5, add_train_ind=False, x_train_val=x_train)
        plt.show()

    # Perform Gaussian process regression for specific kernel
    else:
        mean_prediction, cov_prediction = perform_gpr.perform_gpr_alg(obs_train=data_train_scaled,
                                                                      grid_train=x_train,
                                                                      full_grid=grid_resh.T,
                                                                      kernel=kernel)
        print('R2 score:' + str(r2_score(data_scaled.reshape(np.shape(grid_x1)[0], np.shape(grid_x1)[1]),
                                         mean_prediction.reshape(np.shape(grid_x1)[0], np.shape(grid_x1)[1]))))
        # Plot prediction
        plot_data.make_3D_surface_plot(x=grid_x1, y=grid_x2,
                                       z=mean_prediction.reshape(np.shape(grid_x1)[0], np.shape(grid_x1)[1]))
        # Plot posteriors
        posteriors = plot_data.plot_posteriors(x=grid_x1, y=grid_x2, z=data, mean_pred=mean_prediction.flatten(),
                                               cov_pred=cov_prediction,
                                               posterior_nums=5, add_train_ind=False, x_train_val=x_train)
        plt.show()

    # plot_data.plot_posteriors_squared_error(x=grid_x1, y=grid_x2, z=data, posteriors=posteriors)


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings('ignore')

    perform_1D_gpr(make_grid_search=False, kernel=gp.kernels.ExpSineSquared(length_scale=0.1, length_scale_bounds=(1e-5, 1e5)))
    #perform_2D_gpr(make_grid_search=True)
