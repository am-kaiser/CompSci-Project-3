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