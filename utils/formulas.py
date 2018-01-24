import math
import numpy as np


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


def get_logit_predict(w, X):
    const_col = X.sum(1)[...,None]
    X = np.append(X, const_col, 1)

    return np.array([logit(y) for y in np.dot(X, w)])

