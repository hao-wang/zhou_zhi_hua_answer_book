"""
编程实现基于基尼指数进行划分选择的决策树算法。为西瓜数据集2.0生成预剪枝、后剪枝决策树，并与未剪枝进行对比。

We start with a more comprehensive code based on 4_3.py; we add a keyed
parameter 'method' to function get_best_divider(), in order to choose between
Information gain method and Gini index method.

Author: Hao Wang

"""
import numpy as np
import os
import pandas as pd
from sklearn.cross_validation import train_test_split
import sys

sys.path.append('../utils')
from constants import discrete_cols, continuous_cols_name, continuous_cols
from data_preprocess import get_discretized_values
from formulas import get_gini, get_entropy, get_purity, get_splitted_purity

def get_best_divider(df, cols, purity_rule = 'gini'):
    """
    Get best divider according to information gain.

    Input:
        df: dataframe
        cols: set((column_name, value))

    Return:
        best_divider: (column_name, value)
    """
    min_value = np.inf
    for col in cols:
        if col in continuous_cols:
            prty = get_splitted_purity(df, col, discrete=False,
                    purity_rule=purity_rule)
        else:
            prty = get_splitted_purity(df, col, discrete=True, 
                    purity_rule=purity_rule)

        if prty < min_value:
            min_value = prty
            best_divider = col

    return best_divider


def get_lr_col(df):
    """
    Input:
        df: pd.DataFrame

    Return:
        
    """
    X = df[continuous_cols_name].values
    y = df['好瓜'].values[:]
    y[y=='是'] = 1
    y[y=='否'] = -1

    w = find_best_model(X, y, 0.2, 2000, True)
    y_predict = get_logit_predict(X, w)

    str_col = (str(w[0])+'*'+continuous_cols_name[0] +
            str(w[1])+'*'+continuous_cols_name[1] +
            str(w[2]))
    df[str_col] = y_predict
    return (str_col, -99999), df


def get_best_divider_with_lr(df, cols, purity_rule):
    lr_col, df_tmp = get_lr_col(df)
    best_divider = get_best_divider(df_tmp, df_tmp.columns, purity_rule)
    if best_divider[0] != str_col:
        return best_divider, df, cols
    else:
        return best_divider, df_tmp, cols+[lr_col]


def is_parameters_uniform(df, cols):
    """
    All values on all columns are the same.

    Input:
        df: dataframe
        cols: set((column_name, value))

    Return:
        is_parameters_uniform: Bool
    """
    uniform = True
    for col in cols:
        if len(set(df[col[0]])) > 1:
            return False

    return uniform


def get_majority(df, y_col = '好瓜', prefer_value='是'):
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
    if '是' == prefer_value:
        return ('是' if n_positive>=n_negative else '否')
    else:
        return ('是' if n_positive>n_negative else '否')


def split_makes_worse(df, split_col, discrete=True, purity_rule='gini'):
    func = get_purity(purity_rule)

    base_purity = func(df)
    splitted_purity = get_splitted_purity(df, split_col, discrete, purity_rule)

    if splitted_purity >= base_purity:
        return True
    else:
        return False


def construct_tree(df, cols, y_col = '好瓜', purity_rule = 'gini',
        prune='no', df_test=set(), with_lr='False'):
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
        return list(df[y_col])[0]
    elif len(cols) == 0 or is_parameters_uniform(df, cols):
        reason = "residual" if len(cols)==0 else "majority"
        the_majority = get_majority(df)
        print("Leaf Node(%s) of %s, including %s" % (reason, the_majority, list(df.index)))
        #print(df)
        return the_majority
    else:
        if with_lr == 'True':
            bd, water_melon, cols = get_best_divider_with_lr(df, cols, purity_rule)
        else:
            bd = get_best_divider(df, cols, purity_rule)

        if prune=='pre' and split_makes_worse(df_test, bd, (bd not in continuous_cols), purity_rule):
            the_majority = get_majority(df)
            print("Leaf Node(majority) of %s, including %s" % (the_majority,
                list(df.index)))
            return the_majority

        my_tree = {bd[0]:{}}

        #print("New Node created: ", bs)
        #print(df)

        if bd in continuous_cols:
            df_lt = df[df[bd[0]] < bd[1]]
            df_lt_test = df_test[df_test[bd[0]] < bd[1]]
            print("New branch: %s < %s" % (bd[0], bd[1]))
            my_tree[bd[0]]['<'+str(v)] = construct_tree(
                    df_lt, 
                    set(cols)-{bd}, 
                    purity_rule=purity_rule, 
                    prune=prune,
                    df_test=df_lt_test)

            df_gt = df[df[bd[0]] > bd[1]]
            df_gt_test = df_test[df_test[bd[0]] > bd[1]]
            print("New branch: %s > %s" % (bd[0], bd[1]))
            my_tree[bd[0]]['>'+str(v)] = construct_tree(
                    df_gt, 
                    set(cols)-{bd},
                    purity_rule=purity_rule,
                    prune=prune,
                    df_test=df_gt_test)

        else:
            for v, g in df.groupby(bd[0]):
                print("New branch: %s=%s" % (bd[0], v))
                g_test = df_test[df_test[bd[0]]==v]
                my_tree[bd[0]][v] = construct_tree(
                        g, 
                        set(cols)-{bd}, 
                        purity_rule=purity_rule, 
                        prune=prune,
                        df_test=g_test)

            the_majority = get_majority(df)
            my_tree[bd[0]]['default'] = the_majority
            print("Default branch of %s: " % bd[0], the_majority)

        return my_tree


def get_decision_from_tree(data, tree):
    """
    Get labels for test dataset using the built tree.

    Input: 
        data: pd.Series
        tree: dict (decision tree)
    Return:
        decision: String 
    """

    if type(tree).__name__ == 'dict':
        k = list(tree.keys())[0]
        data_value = data[k]
        #print(k, data_value)
        return get_decision_from_tree(data, tree[k].get(data_value, 'default'))

    else:
        return tree


if __name__ == '__main__':
    melon_file = os.path.abspath('../data/xigua_2.0.csv')
    water_melon = pd.read_csv(melon_file, index_col='编号')

    train, validate = train_test_split(water_melon, train_size=0.6,
            random_state=1)
    print(len(train), len(validate))

    # Unify the format of discrete/continuous columns to be (name, split_value).
    try:
        rho = get_discretized_values(list(train.密度))
        sugar = get_discretized_values(list(train.含糖率))
        continuous_cols = list(zip(['密度']*len(rho), rho)) + list(zip(['含糖率']*len(sugar), sugar))
    except Exception as e:
        continuous_cols = []

    all_cols = discrete_cols  + continuous_cols

    tmp_cols = all_cols.copy()
    print("no_prune"+"="*50)
    no_prune_tree = construct_tree(train, tmp_cols, purity_rule='entropy', prune='no', df_test=validate)

    print("pre_prune"+"="*50)
    tmp_cols = all_cols.copy()
    pre_prune_tree = construct_tree(train, tmp_cols, purity_rule = 'gini', prune='pre', df_test=validate)

    print('no_prune: ', no_prune_tree)
    print('pre_prune: ', pre_prune_tree)

    print(type(validate))
    for idx, s in validate.iterrows():
        #print(s)
        decision_1 = get_decision_from_tree(s, no_prune_tree)
        decision_2 = get_decision_from_tree(s, pre_prune_tree)
        #print(decision_1, decision_2)
        #print("="*50)

    #print("pre_prune="*50)
    #tmp_cols = all_cols.copy()
    #construct_tree(train, tmp_cols, purity_rule = 'gini', prune='pre', df_test=validate)
