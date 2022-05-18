import matplotlib.pyplot as plt


def make_2d_plot(data, grid, y_label, file_name):
    # Create plot
    plt.figure(figsize=(5, 5))
    plt.rc('font', **{'size': '11'})
    plt.plot(grid, data)
    plt.xlabel('x')
    plt.ylabel(y_label)
    #plt.savefig(file_name, bbox_inches='tight')
    plt.show()


def make_conf_interval_plot(x_values, y_values, x_values_train, y_values_train, x_values_test, mean_pred, std_pred):
    plt.plot(x_values, y_values, linestyle="dotted")
    plt.scatter(x_values_train, y_values_train, label="Observations")
    plt.scatter(x_values_test, mean_pred, label="Mean Prediction", marker='v')
    plt.fill_between(
        x_values_test.ravel(),
        mean_pred - 1.95 * std_pred,
        mean_pred + 1.95 * std_pred,
        alpha=0.7,
        label=r"95% confidence interval",
    )
    plt.legend()
    plt.xlabel("$x$")
    plt.ylabel("$f(x)$")
    plt.show()