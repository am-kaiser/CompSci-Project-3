from gpr_alg import data_def, plot_data

# Make grid
grid = data_def.create_1D_grid(step_size=0.01)

# Create data
data = data_def.create_data(func_param=[1, grid], func_name='zhouetal')

# Plot data
plot_data.make_2d_plot(data, grid, y_label='f(x)', file_name='data_zhouetal.png')
