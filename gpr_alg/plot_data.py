import matplotlib.pyplot as plt
import numpy as np
# from cmcrameri import cm
import scipy.stats as st


def make_2D_plot(data, grid, y_label):
    # Create plot
    plt.rc('font', **{'size': '11'})
    fig, ax = plt.subplots(figsize=(5, 5))
    fig.tight_layout(pad=3)
    ax.plot(grid, data)
    ax.set_xlabel('x')
    ax.set_ylabel(y_label)


def make_3D_surface_plot(x, y, z, file_name=None):
    """Create 3D surface plot of given input."""
    # Create figure and axes for plot
    plt.rc('font', **{'size': '11'})
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(5, 5))
    fig.tight_layout(pad=3)
    # Plot the surface.
    surf = ax.plot_surface(x, y, z, linewidth=0, antialiased=False)  # , cmap=cm.batlow)
    # Customize the z axis.
    ax.set_xlabel('x')
    ax.set_ylabel('t')
    ax.set_zlabel('u(x,t)')
    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.4, aspect=5)
    ax.view_init(21, 50)
    if file_name is not None:
        fig.savefig(file_name, bbox_inches='tight')


def make_3D_contour_plot(x, y, z, add_train=False, x_train=None, y_train=None, file_name=None):
    """Create 3D contour plot of given input."""
    # Create figure and axes for plot
    plt.rc('font', **{'size': '11'})
    fig, ax = plt.subplots(figsize=(5, 5))
    fig.tight_layout(pad=3)
    # Plot the surface.
    surf = ax.contourf(x, y, z)
    # Add train points
    if add_train:
        ax.scatter(x_train, y_train, color='red')
    # Customize the z axis.
    ax.set_xlabel('x')
    ax.set_ylabel('t')
    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)
    if file_name is not None:
        fig.savefig(file_name, bbox_inches='tight')


def make_2D_conf_interval_plot(grid_values, obs_values, grid_values_train, obs_values_train, mean_pred, std_pred):
    plt.rc('font', **{'size': '11'})
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.tight_layout(pad=3)
    ax.plot(grid_values, obs_values, linestyle="dotted")
    ax.scatter(grid_values_train, obs_values_train, label="Observations", s=50)
    ax.scatter(grid_values, mean_pred, label="Mean Prediction", marker='v')
    ax.fill_between(
        grid_values.ravel(),
        mean_pred - 1.95 * std_pred,
        mean_pred + 1.95 * std_pred,
        alpha=0.7,
        label=r"95% confidence interval",
    )
    ax.legend()
    ax.set_xlabel("$x$")
    ax.set_ylabel("$f(x)$")


def plot_posteriors(x, y, z, mean_pred, cov_pred, posterior_nums=5, add_train_ind=False, x_train_val=None,
                    file_name=None):
    # Calculate posteriors
    posteriors = st.multivariate_normal.rvs(mean=mean_pred, cov=cov_pred, size=posterior_nums)

    # To have same color range for all subplots find min and max value
    min_val = np.nanmin(np.array(np.nanmin(posteriors), np.nanmin(z)))
    max_val = np.nanmax(np.array(np.nanmax(posteriors), np.nanmax(z)))
    levels = np.round(np.linspace(min_val, max_val, 10), 2)

    # Plot analytic solution
    plt.rc('font', **{'size': '11'})
    fig, axs = plt.subplots(int(np.ceil((posterior_nums + 1) / 2)), 2, figsize=(5, 5))
    fig.tight_layout(pad=3)
    axs = axs.ravel()
    ax = axs[0]
    ax.set_xlabel('x')
    ax.set_ylabel('t')
    cs = ax.contourf(x, y, z, levels=levels, extend='min')
    # Add indicators for train dataset
    if add_train_ind:
        ax.plot(x_train_val[:, 0], x_train_val[:, 1], "r.", ms=10)

    # Plot posteriors
    for i, post in enumerate(posteriors, 1):
        cs = axs[i].contourf(x, y, post.reshape(-1, np.shape(x)[0]), levels=levels, extend='min')
        axs[i].set_xlabel('x')
        axs[i].set_ylabel('t')

    # Place colorbar at desire position
    fig.colorbar(cs, ax=axs.ravel().tolist(), shrink=0.95)

    if file_name is not None:
        fig.savefig(file_name, bbox_inches='tight')

    return posteriors


def plot_posteriors_squared_error(x, y, z, posteriors, single=False):
    if single:
        # Create figure
        fig, axs = plt.subplots(int(np.ceil((np.shape(posteriors)[0] + 1) / 2)), 2, figsize=(5, 5))
        axs = axs.ravel()

        # Plot squared error for every posterior
        for i, post in enumerate(posteriors, 1):
            squared_error = (z - post.reshape(-1, np.shape(x)[0])) ** 2
            cs = axs[i].contourf(x, y, squared_error)
            fig.colorbar(cs, ax=axs[i], shrink=0.95)
    else:
        squared_error = np.zeros(np.shape(z))
        # calculate mean squared error over all posteriors
        for i, post in enumerate(posteriors, 1):
            squared_error += (z - post.reshape(-1, np.shape(x)[0])) ** 2

        squared_error = squared_error / np.shape(posteriors)[0]

        fig, axs = plt.subplots(1)
        cs = axs.contourf(x, y, squared_error)
        fig.colorbar(cs, ax=axs, shrink=0.95)
