"""
编程实现基于信息熵进行划分选择的决策树算法。为西瓜数据集3.0生成决策树。

Author: Hao Wang
"""
import numpy as np
import os
import pandas as pd

def get_discretized_values(lst):
    """
    Discretize continuous parameters by choosing splits.

    Input:
        lst: [value_of_continuous_column]

    Return:
        splits: [value_of_splits]
    """
    try:
        lst = [float(ele) for ele in lst]
    except Exception as e:
        print(e)
        return ['-']
    #assert(all([float==type(ele) for ele in lst]))

    lst = list(set(lst))
    sort = sorted(lst)
    return [(sort[i]+sort[i+1])/2 for i in range((len(sort)-1))]


def get_base_entropy(df, y_col='好瓜'):
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


def get_splitted_entropy(df, split_col):
    """
    Get the total information entropy of splitted datasets.

    Input:
        df: dataframe
        split_col: (column_name, -9999)

    Return:
        information_entropy: float
    """

    tot_ent = 0
    IV = 0
    N = len(df)
    split_col = split_col[0]
    for v, g in df.groupby(split_col):
        ent = get_base_entropy(g)
        n = len(g)
        tot_ent += (n/N) * ent
        IV += -(n/N) * np.log2(n/N)

    return tot_ent/IV


def get_splitted_entropy_continuous(df, col):
    """
    Get information entropy for splitted data (divide the data by two
    according to the value of the splitting column).

    Input:
        df: dataframe
        col: (column_name, splitting_value)

    Return:
        information_entropy: float
    """

    N = len(df)
    split_col, split = col[0], col[1]

    tot_ent = 0
    df_lt = df[df[split_col] < split]
    df_gt = df[df[split_col] > split]
    n_lt = len(df_lt)
    n_gt = len(df_gt)
    tot_ent = ((n_lt/N)*get_base_entropy(df_lt) +
               (n_gt/N)*get_base_entropy(df_gt))

    return tot_ent


def get_best_divider(df, cols):
    """
    Get best divider according to information gain.

    Input:
        df: dataframe
        cols: set((column_name, value))

    Return:
        best_divider: (column_name, value)
    """
    max_inc = -1
    for col in cols:

        if col in continuous_cols:
            entropy_inc = (get_base_entropy(df) - get_splitted_entropy_continuous(df, col))
        else:
            entropy_inc = (get_base_entropy(df) - get_splitted_entropy(df, col))

        if max_inc < entropy_inc:
            max_inc = entropy_inc
            best_divider = col

    return best_divider


def is_uniform(df, cols):
    """
    All values on all columns are the same.

    Input:
        df: dataframe
        cols: set((column_name, value))

    Return:
        is_uniform: Bool
    """
    uniform = True
    for col in cols:
        if len(set(df[col[0]])) > 1:
            return False

    return uniform


def get_majority(df, y_col = '好瓜'):
    """
    Get the majority of the y_value.

    Input: 
        df: dataframe
        y_col: String

    Return:
        majority of y_col: String
    """
    n_positive = len(df[df[y_col]=='是'])
    n_negative = len(df[df[y_col]=='否'])
    return ('是' if n_positive>=n_negative else '否')


def construct_tree(df, cols, y_col = '好瓜'):
    """
    The main function.

    Input:
        df: dataframe
        cols: set((column_name, value))
        y_col: String

    Return:
        None
    """
    if len(set(df[y_col])) == 1:
        print("Leaf Node(pure) of %s, including %s" % (list(df[y_col])[0], list(df.index)))
        #print(df)
        return
    elif len(cols) == 0 or is_uniform(df, cols):
        reason = "residual" if len(cols)==0 else "uniform"
        the_majority = get_majority(df)
        print("Leaf Node(%s) of %s, including %s" % (reason, the_majority, list(df.index)))
        #print(df)
        return
    else:
        bs = get_best_divider(water_melon, cols)

        #print("Create node: ", bs)
        if bs in continuous_cols:
            df_lt = df[df[bs[0]] < bs[1]]
            print("New branch: %s < %s" % (bs[0], bs[1]))
            construct_tree(df_lt, set(cols)-{bs})

            df_gt = df[df[bs[0]] > bs[1]]
            print("New branch: %s > %s" % (bs[0], bs[1]))
            construct_tree(df_gt, set(cols)-{bs})

        else:
            for v, g in df.groupby(bs[0]):
                print("New branch: %s=%s" % (bs[0], v))
                construct_tree(g, set(cols)-{bs})

if __name__ == '__main__':
    melon_file = os.path.abspath('../data/xigua_3.0.csv')
    water_melon = pd.read_csv(melon_file, index_col='编号')

    # Unify the format of discrete/continuous columns to be (name, split_value).
    rho = get_discretized_values(list(water_melon.密度))
    sugar = get_discretized_values(list(water_melon.含糖率))
    continuous_cols = list(zip(['密度']*len(rho), rho)) + list(zip(['含糖率']*len(sugar), sugar))

    discrete_cols_name = ['色泽', '根蒂', '敲声', '纹理', '脐部', '触感']
    discrete_cols = list(zip(discrete_cols_name, [-99999]*len(discrete_cols_name)))

    all_cols = discrete_cols  + continuous_cols

    construct_tree(water_melon, all_cols)

