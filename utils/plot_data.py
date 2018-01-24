import matplotlib.pyplot as plt
import numpy as np


def plot_scatter(ax, coords, color='b', marker='+'):
    """
    Input:
        ax:
        coords: 

    Return:
        Axes
    """
    x = coords[:, 0]
    y = coords[:, 1]
    ax.scatter(x, y, c=color, marker=marker)
    return ax


def plot_line(ax, k, origin=(0, 0), x_range=(0, 1)):
    """
    Input:
        ax:
        direct_vec: 
        origin:

    Return:
        Axes
    """
    b = origin[1] - k*origin[0]
    
    xs = np.linspace(x_range[0], x_range[1])
    ys = k*xs + b
    ax.plot(xs, ys)

    return ax

