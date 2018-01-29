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


def get_logit_predict(X, w):
    """
    Input:
        w: List OR np.array(n)
        X: np.array(m, n)

    Return:
        y_pred: np.array(m)
    """
    const_col = X.sum(1)[...,None]
    X = np.append(X, const_col, 1)
    X[:, -1] = 1

    return np.array([logit(y) for y in np.dot(X, w)])


def get_gini(df, y_col = '好瓜'):
    """
    Gini index is the probability of randomly picked
    two elements to have different y_col value.

    Input:
        df: dataframe
        y_col: String

    Return:
        Gini_index: float
    """
    print("inside get_gini: ", df)
    cnt_dict = df[y_col].value_counts()
    cnt_good = cnt_dict.get('是', 0)
    cnt_bad = cnt_dict.get('否', 0)
    p_good = cnt_good/(cnt_good + cnt_bad)
    p_bad = cnt_bad/(cnt_good + cnt_bad)

    return 1 - p_good*p_good - p_bad*p_bad


def get_entropy(df, y_col='好瓜'):
    """
    Get information entropy for data.

    Input:
        df: dataframe
        y_col: y value of dataframe

    Return:
        information_entropy: float
    """
    N = len(df)
    dct = dict(df[y_col].value_counts())
    n_positive = dct.get('是', 0)
    n_negative = dct.get('否', 0)

    if n_positive == 0:
        #print("all good now.")
        ent = -(n_negative/N)*np.log2(n_negative/N)
    elif n_negative == 0:
        #print("all bad now.")
        ent = -(n_positive/N)*np.log2(n_positive/N)
    else:
        ent = -(n_negative/N)*np.log2(n_negative/N)-(n_positive/N)*np.log2(n_positive/N)
    return ent


def get_purity(purity_rule):
    if purity_rule == 'gini':
        return get_gini
    else:
        return get_entropy


def get_splitted_purity(df, split_col, discrete=True, purity_rule='entropy'):
    """
    Get the total information entropy of splitted datasets.

    Input:
        df: dataframe
        split_col(discrete): (column_name, -9999)
        split_col(continuous): (column_name, splitting_value)

    Return:
        information_entropy: float
    """
    tot_purity = 0
    N = len(df)
    func = get_purity(purity_rule)

    if discrete:
        #IV = 0
        col = split_col[0]
        for v, g in df.groupby(col):
            prty = func(g)
            n = len(g)
            tot_purity += (n/N) * prty
            #IV += -(n/N) * np.log2(n/N)

        return tot_purity #/IV

    else:
        col, split = split_col[0], split_col[1]

        df_lt = df[df[col] < split]
        df_gt = df[df[col] > split]
        n_lt = len(df_lt)
        n_gt = len(df_gt)
        tot_purity = ((n_lt/N)*func(df_lt) +
                   (n_gt/N)*func(df_gt))

        return tot_purity


