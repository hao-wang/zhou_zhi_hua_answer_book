"""
编程实现线性判别分析。

Author: Wang Hao
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys

sys.path.append('../utils')
from plot_data import plot_scatter, plot_line
from p_3_3_logistic_regression import logit, find_best_model, get_logit_predict


def get_project_vector(X1, X2):
    """
    Input:
        X1: class-1 data, m length-n arrays; np.array(m, n)
        X2: class-2 data, m' length-n arrays; np.array(m', n)
        m1: mean vector of X1; np.array(1, n)
        m2: mean vector of X2; np.array(1, n)

    Return:
        Projecting vector; np.array(1, n)
    """

    def get_Sw(X1, X2):
        """
        Given data arrays and the mean, return Sw, the "covariance".

        Return: Within-class scatter matrix; np.array(n, n)
        """
        n_X1 = X1.shape[0]
        m1 = np.mean(X1, axis=0)
        m1_matrix = np.ones([n_X1, 1]) * m1
        res_X1 = X1-m1_matrix
        #print(m1_matrix)

        n_X2 = X2.shape[0]
        m2 = np.mean(X2, axis=0)
        m2_matrix = np.ones([n_X2, 1]) * m2
        res_X2 = X2-m2_matrix

        return m1, m2, np.dot(res_X1.T, res_X1) + np.dot(res_X2.T, res_X2)

    m1, m2, Sw = get_Sw(X1, X2)
    proj_vector = np.dot(np.linalg.inv(Sw), (m1-m2))
    proj_vector = proj_vector/np.sqrt(np.dot(proj_vector, proj_vector))

    return proj_vector


def plot_result(good_melon, bad_melon):
    all_data = np.concatenate((good_melon, bad_melon), axis=0)
    x1_range = (min(all_data[:, 0]), max(all_data[:, 0]))
    x2_range = (min(all_data[:, 1]), max(all_data[:, 1]))
    center = np.mean(all_data, axis=0)

    fig, ax = plt.subplots()
    ax = plot_scatter(ax, good_melon, 'r', 'o')
    ax = plot_scatter(ax, bad_melon, 'b', '+')
    ax = plot_line(ax, proj_good[1]/proj_bad[0], center, x_range = x1_range)
    plt.ylim(x2_range)

    fig.savefig('melon.png')
    plt.close(fig)


if __name__ == '__main__':
    #df = pd.read_csv('../data/xigua_3.0_alpha.csv')
    df = pd.read_csv('../data/xigua_3.0_fake.csv')
    df_array = df.values
    
    # Group by is_haogua.
    good_idx = (df_array[:, 2] == 1)
    good_melon = df_array[good_idx][:, :2]
    bad_melon = df_array[~good_idx][:, :2]

    proj_vector = get_project_vector(good_melon, bad_melon)

    # Get 1-D 'training' data.
    # Use Logistic regression to find the threshold.    
    proj_good = np.dot(good_melon, proj_vector)
    proj_bad = np.dot(bad_melon, proj_vector)
    print('good: ', proj_good)
    print('bad: ', proj_bad)

    X = np.concatenate([np.array(proj_good), np.array(proj_bad)])[np.newaxis].T
    y = len(proj_good) * [1] + len(proj_bad) * [0]
    w = find_best_model(X, y, 0.1, 300)
    print("best model: ", w)

    y_pred = get_logit_predict(w, X)
    df['pred'] = y_pred
    print(df)

    # Check out the result.
    plot_result(good_melon, bad_melon)
