"""
编程实现对率回归；给出西瓜数据集3.0_alpha的结果。

Author: Hao Wang

Task: 识别西瓜好坏
Performance: 判断结果的准确率
Experience: 西瓜数据集3.0_alpha
"""

import numpy as np
import pandas as pd
import scipy.optimization


def logit(x, thresh = 0.5):
    return 1 if (1/(1+math.exp(-x))>thresh) else 0

def loss_rms(X, y, w, b):
    """
    Input:
        X: np.array([M, N])
        y: np.array(M)
        w: np.array(N)
        b: float

    Return: 
        loss: float
    """
    loss = sum([(logit(_)-answer)*(logit(_)-answer) 
        for (_, answer) in zip(np.dot(X, w) + b, y)])

    return loss
   
def find_best_model(df, y_col = '好瓜'):
    X = df.loc[:, df.columns != y_col].values
    y = df[y_col].values

    n_params = len(df.columns) - 1
    p = {'w':np.ones(n_params), 'b':0., 'X':X, 'y'=y}
    best = find_minimum(loss_rms, **p)
    return best
