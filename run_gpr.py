from gpr_alg import prepare_data, plot_data, perform_gpr

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score


def perform_1D_gpr():
    # Make grid
    grid = prepare_data.create_1D_grid(step_size=0.01)
    # Create data
    data = prepare_data.create_data(func_param=[1, grid], func_name='zhouetal')
    # Plot data
    plot_data.make_2D_plot(data, grid, y_label='f(x)', file_name='data_zhouetal.png')
    # Split data into test and training datasets
    grid_train, grid_test, data_train, data_test = prepare_data.split_train_test_data(grid, data, train_perc=0.1)
    # # Perform Gaussian process regression
    # mean_prediction, std_prediction = perform_gpr.perform_gpr_alg(obs_train=data_train, grid_train=grid_train,
    #                                                               full_grid=grid, get_std=True)
    # perform grid search for best model
    mean_prediction, std_prediction = perform_gpr.search_best_model(obs_train=data_train, grid_train=grid_train,
                                                                    full_grid=grid, get_std=True)

    plot_data.make_2D_conf_interval_plot(grid_values=grid, obs_values=data, grid_values_train=grid_train,
                                         obs_values_train=data_train, mean_pred=mean_prediction.flatten(),
                                         std_pred=std_prediction.flatten())
    plt.show()


def perform_2D_gpr():
    # Make grid
    # Note: x1=x, x2=t
    grid_x1, grid_x2 = prepare_data.create_2D_grid(x1_step_size=0.1, x2_step_size=0.1)

    # Reshape grid values
    grid_resh = np.stack([grid_x1.flatten(), grid_x2.flatten()])

    # Split data into test and training datasets
    size_train_set = int(np.shape(grid_x1.flatten())[0] * 0.5)
    x_train = np.stack(
        (np.random.choice(grid_x1.flatten(), size_train_set), np.random.choice(grid_x2.flatten(), size_train_set)),
        axis=-1)

    # Create data
    data_train = prepare_data.create_data(func_param=[x_train[:, 0], x_train[:, 1]], func_name='heat_equation')
    data = prepare_data.create_data(func_param=[grid_x1, grid_x2], func_name='heat_equation')

    # Plot data
    plot_data.make_3D_surface_plot(x=grid_x1, y=grid_x2, z=data)

    # Scale data (i.e. normalize)
    data_train_scaled = prepare_data.rescale_data(data_train.reshape(-1, 1))

    # # search For best model with a grid search
    # mean_prediction, cov_prediction = perform_gpr.search_best_model(obs_train=data_train_scaled, grid_train=x_train,
    #                                                                 full_grid=grid_resh.T)

    # Perform Gaussian process regression for specific kernel
    mean_prediction, cov_prediction = perform_gpr.perform_gpr_alg(obs_train=data_train_scaled,
                                                                  grid_train=x_train,
                                                                  full_grid=grid_resh.T)
    print('R2 score:' + str(r2_score(data, mean_prediction.reshape(np.shape(grid_x1)[0], np.shape(grid_x1)[1]))))

    # Plot prediction
    plot_data.make_3D_surface_plot(x=grid_x1, y=grid_x2,
                                   z=mean_prediction.reshape(np.shape(grid_x1)[0], np.shape(grid_x1)[1]))

    # Plot posteriors
    posteriors = plot_data.plot_posteriors(x=grid_x1, y=grid_x2, z=data, mean_pred=mean_prediction.flatten(),
                                           cov_pred=cov_prediction,
                                           posterior_nums=5, add_train_ind=True, x_train_val=x_train)

    # plot_data.plot_posteriors_squared_error(x=grid_x1, y=grid_x2, z=data, posteriors=posteriors)

    plt.show()


if __name__ == "__main__":
    perform_1D_gpr()
    # perform_2D_gpr()
