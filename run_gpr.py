from gpr_alg import data_def, plot_data, perform_gpr

import numpy as np


# Make grid
grid = data_def.create_1D_grid(step_size=0.2)
grid_2 = data_def.create_1D_grid(step_size=0.01)

# Create data
data = data_def.create_data(func_param=[1, grid], func_name='zhouetal')

# Plot data
#plot_data.make_2d_plot(data, grid, y_label='f(x)', file_name='data_zhouetal.png')

# Split data into test and training datasets
grid_train, grid_test, data_train, data_test = data_def.split_train_test_data(grid, data)

# Perform Gaussian process regression
mean_prediction, std_prediction = perform_gpr.define_gpr_alg(data_train, grid_train, grid_2)
#print(mean_prediction, std_prediction)

plot_data.make_conf_interval_plot(grid, data, grid_train, data_train, grid_2, mean_prediction, std_prediction)