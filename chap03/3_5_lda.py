"""
编程实现线性判别分析。

Author: Wang Hao
"""
import numpy as np
import pandas as pd


def get_project_vector(X1, X2, m1, m2):
    """
    Input:
        X1: class-1 data, m length-n arrays; np.array(m, n)
        X2: class-2 data, m length-n arrays; np.array(m, n)
        m1: mean vector of X1; np.array(1, n)
        m2: mean vector of X2; np.array(1, n)

    Return:
        Projecting vector; np.array(1, n)
    """

    def get_Sw(X1, X2, m1, m2):
        """
        Given data arrays and the mean, return Sw, the "covariance".

        Return: Within-class scatter matrix; np.array(n, n)
        """
        n_X1 = X1.shape[0]
        m1_matrix = np.ones([n_X1, 1]) * m1
        n_X2 = X2.shape[0]
        m2_matrix = np.ones([n_X2, 1]) * m2

        res_X1 = X1-m1_matrix
        res_X2 = X2-m2_matrix

        return np.dot(res_X1.T, res_X1) + np.dot(res_X2.T, res_X2)


    sw = get_Sw(X1, X2, m1, m2)
    proj_vector = np.dot(np.linalg.inv(sw), (m1-m2))
    proj_vector = proj_vector/np.sqrt(np.dot(proj_vector, proj_vector))

    return proj_vector


if __name__ == '__main__':
    df = pd.read_csv('../data/xigua_3.0_alpha.csv')

    df_good = df[df.好瓜==1]
    df_bad = df[df.好瓜==0]

    good_melon = df_good.loc[:, df_good.columns != '好瓜'].values
    bad_melon = df_bad.loc[:, df_bad.columns != '好瓜'].values

    m_good = np.mean(good_melon, axis=0)
    #print(m_good)
    m_bad = np.mean(bad_melon, axis=0)

    proj_vector = get_project_vector(good_melon, bad_melon, m_good, m_bad)
    proj_good = np.dot(good_melon, proj_vector)
    proj_bad = np.dot(bad_melon, proj_vector)

    mean_good, std_good = np.mean(proj_good), np.std(proj_good)
    mean_bad, std_bad = np.mean(proj_bad), np.std(proj_bad)

    print("metrics of good and bad: ", mean_good, std_good, mean_bad, std_bad)

    # Make it a rule that mean_bad > mean_good
    if mean_bad < mean_good:
        proj_vector = -proj_vector

    proj_good = np.dot(good_melon, proj_vector)
    mean_good, std_good = np.mean(proj_good), np.std(proj_good)
    mean_bad, std_bad = np.mean(proj_bad), np.std(proj_bad)

    proj_thresh = min([mean_good+1*std_good, mean_bad-1*std_good])

    y_pred = (np.dot(df.loc[:, df.columns != '好瓜'].values, proj_vector) > proj_thresh)
    df['pred'] = y_pred
    print(df)
