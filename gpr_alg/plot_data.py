import matplotlib.pyplot as plt
#from cmcrameri import cm
import scipy.stats as st
import numpy as np


def make_2D_plot(data, grid, y_label, file_name):
    # Create plot
    plt.figure(figsize=(5, 5))
    plt.rc('font', **{'size': '11'})
    plt.plot(grid, data)
    plt.xlabel('x')
    plt.ylabel(y_label)
    # plt.savefig(file_name, bbox_inches='tight')
    #plt.show()


def make_3D_surface_plot(x, y, z, title=None, fig=None, position=111, add_colorbar=True):
    """Create 3D surface plot of given input."""
    # Create figure and axes for plot
    if not fig:
        fig = plt.figure()
    ax = fig.add_subplot(position, projection='3d')
    # Plot the surface.
    surf = ax.plot_surface(x, y, z, linewidth=0, antialiased=False)#, cmap=cm.batlow)
    # Customize the z axis.
    ax.set_xlabel('x')
    ax.set_ylabel('t')
    ax.set_zlabel('u(x,t)')
    # Add a color bar which maps values to colors.
    if add_colorbar:
        fig.colorbar(surf, shrink=0.5, aspect=5)
    if title:
        fig.suptitle(title)
    #plt.show()


def make_2D_conf_interval_plot(grid_values, obs_values, grid_values_train, obs_values_train, mean_pred, std_pred):
    plt.plot(grid_values, obs_values, linestyle="dotted")
    plt.scatter(grid_values_train, obs_values_train, label="Observations", s=50)
    plt.scatter(grid_values, mean_pred, label="Mean Prediction", marker='v')
    plt.fill_between(
        grid_values.ravel(),
        mean_pred - 1.95 * std_pred,
        mean_pred + 1.95 * std_pred,
        alpha=0.7,
        label=r"95% confidence interval",
    )
    plt.legend()
    plt.xlabel("$x$")
    plt.ylabel("$f(x)$")
    #plt.show()


def plot_posteriors(x, y, z, mean_pred, cov_pred, posterior_nums=5, add_train_ind=False, x_train_val=None):
    # Calculate posteriors
    posteriors = st.multivariate_normal.rvs(mean=mean_pred, cov=cov_pred,size=posterior_nums)

    # To have same color range for all subplots find min and max value
    min_val = np.nanmin(np.array(np.nanmin(posteriors), np.nanmin(z)))
    max_val = np.nanmax(np.array(np.nanmax(posteriors), np.nanmax(z)))
    levels = np.round(np.linspace(min_val, max_val, 10),2)

    # Plot analytic solution
    fig, axs = plt.subplots(int(np.ceil((posterior_nums + 1)/2)), 2)
    axs = axs.ravel()
    ax = axs[0]
    cs = ax.contourf(x, y, z, levels=levels, extend='min')
    # Add indicators for train dataset
    if add_train_ind:
        ax.plot(x_train_val[:, 0], x_train_val[:, 1], "r.", ms=10)
    #fig.colorbar(cs, ax=ax)

    # Plot posteriors
    for i, post in enumerate(posteriors, 1):
        cs = axs[i].contourf(x, y, post.reshape(-1, np.shape(x)[0]), levels=levels, extend='min')

    # Place colorbar at desire position
    fig.colorbar(cs, ax=axs.ravel().tolist(), shrink=0.95)

    return posteriors


def plot_posteriors_squared_error(x, y, z, posteriors, single=False):
    if single:
        # Create figure
        fig, axs = plt.subplots(int(np.ceil((np.shape(posteriors)[0] + 1) / 2)), 2)
        axs = axs.ravel()

        # Plot squared error for every posterior
        for i, post in enumerate(posteriors, 1):
            squared_error = (z - post.reshape(-1, np.shape(x)[0]))**2
            cs = axs[i].contourf(x, y, squared_error)
            fig.colorbar(cs, ax=axs[i], shrink=0.95)
    else:
        squared_error = np.zeros(np.shape(z))
        # calculate mean squared error over all posteriors
        for i, post in enumerate(posteriors, 1):
            squared_error += (z - post.reshape(-1, np.shape(x)[0]))**2

        squared_error = squared_error / np.shape(posteriors)[0]

        fig, axs = plt.subplots(1)
        cs = axs.contourf(x, y, squared_error)
        fig.colorbar(cs, ax=axs, shrink=0.95)
