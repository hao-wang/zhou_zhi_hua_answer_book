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
        split_col = split_col[0]
        for v, g in df.groupby(split_col):
            prty = func(g)
            n = len(g)
            tot_purity += (n/N) * prty
            #IV += -(n/N) * np.log2(n/N)

        return tot_purity #/IV

    else:
        split_col, split = col[0], col[1]

        df_lt = df[df[split_col] < split]
        df_gt = df[df[split_col] > split]
        n_lt = len(df_lt)
        n_gt = len(df_gt)
        tot_purity = ((n_lt/N)*func(df_lt) +
                   (n_gt/N)*func(df_gt))

        return tot_purity


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
            prty = get_splitted_purity(df, col, False, purity_rule)
        else:
            prty = get_splitted_purity(df, col, True, purity_rule)

        if prty < min_value:
            min_value = prty
            best_divider = col

    return best_divider


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
        prune='no', df_test=set()):
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
        bd = get_best_divider(water_melon, cols, purity_rule)

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


if __name__ == '__main__':
    melon_file = os.path.abspath('../data/xigua_2.0.csv')
    water_melon = pd.read_csv(melon_file, index_col='编号')

    train, validate = train_test_split(water_melon, train_size=0.6,
            random_state=1)
    print(len(train), len(validate))
    input()

    # Unify the format of discrete/continuous columns to be (name, split_value).
    try:
        rho = get_discretized_values(list(train.密度))
        sugar = get_discretized_values(list(train.含糖率))
        continuous_cols = list(zip(['密度']*len(rho), rho)) + list(zip(['含糖率']*len(sugar), sugar))
    except Exception as e:
        continuous_cols = []

    discrete_cols_name = ['色泽', '根蒂', '敲声', '纹理', '脐部', '触感']
    discrete_cols = list(zip(discrete_cols_name, [-99999]*len(discrete_cols_name)))

    all_cols = discrete_cols  + continuous_cols

    tmp_cols = all_cols.copy()
    print("no_prune"+"="*50)
    no_prune_tree = construct_tree(train, tmp_cols, purity_rule='entropy', prune='no', df_test=validate)

    print("pre_prune"+"="*50)
    tmp_cols = all_cols.copy()
    pre_prune_tree = construct_tree(train, tmp_cols, purity_rule = 'gini', prune='pre', df_test=validate)

    print('no_prune: ', no_prune_tree)
    print('pre_prune: ', pre_prune_tree)

    #print("pre_prune="*50)
    #tmp_cols = all_cols.copy()
    #construct_tree(train, tmp_cols, purity_rule = 'gini', prune='pre', df_test=validate)
