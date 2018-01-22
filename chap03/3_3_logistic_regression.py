"""
编程实现对率回归；给出西瓜数据集3.0_alpha的结果。

Author: Hao Wang

Task: 识别西瓜好坏
Performance: 判断结果的准确率
Experience: 西瓜数据集3.0_alpha
"""
import math
import numpy as np
import pandas as pd

def logit(x, thresh = 0.5):
    """
    Input:
        x: float
        thresh: float
    Return:
        logit: 0 or 1
    """
    #return 1 if (1/(1+math.exp(-x))>thresh) else 0
    return (1/(1+math.exp(-x)))>thresh


def loss_cross_entropy(X, y, w):
    """
    Cross entropy as loss. 

    Input:
        X: input data, np.array([M, N])
        y: labels, np.array(M)
        w: parameters, np.array(N)

    Return: 
        loss: float
    """
    loss = 0
    N = X.shape[1]
    for idx in range(X.shape[1]):
        loss += 1./N * np.log(1+np.exp(-y[idx]*np.dot(w, X[idx, :])))
    return loss
   

def loss_grad_wrt_w(X, y, w):
    """
    Input:
        X: np.array 
        y: np.array of labels, -1/1

    Return:
        loss_grad: float
    """
    N = len(X)
    loss_grad = 0
    for i in range(N):
        loss_grad += -(1./(1+np.exp(-y[i] * (np.dot(w, X[i, :])))) * 
                np.exp(-y[i] * (np.dot(w, X[i, :]))) *
                (y[i]*X[i, :]))

    loss = loss_cross_entropy(X, y, w)
    #print("loss: " + str(loss))
    return loss_grad/N


def find_best_model(X, y, w, learn_rate, time_steps):
    for _ in range(time_steps): 
        w = w - loss_grad_wrt_w(X, y, w)*learn_rate
        #print(w)

    return w


if __name__ == '__main__':
    df = pd.read_csv('../data/xigua_3.0_alpha.csv')
    y_col = '好瓜'

    # Add the bias column.
    X = df.loc[:, df.columns != y_col].values
    const_col = X.sum(1)[...,None]
    X = np.append(X, const_col, 1)

    # For using cross-entropy loss function, make the 1/0 binary value to be 1/-1
    y = df[y_col].values
    y[y==0] = -1

    # Find the best parameter using gradient descent
    w = np.ones(X.shape[1])
    w = find_best_model(X, y, w, 0.5, 500)

    df['pred'] = [logit(f) for f in np.dot(X, w)]
    print(df)
