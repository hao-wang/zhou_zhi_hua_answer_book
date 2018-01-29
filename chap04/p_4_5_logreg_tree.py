"""
基于对率回归进行划分选择的决策树算法，为表4.3生成决策树。

Author: Hao Wang
"""
import os
import pandas as pd
import sys
sys.path.append('../chap03')
from p_3_3_logistic_regression import find_best_model
sys.path.append('../utils')
from constants import discrete_cols, continuous_cols_name
from formulas import get_logit_predict
from p_4_4_decision_tree_gini import construct_tree
from formulas import get_gini, get_entropy, get_purity, get_splitted_purity


if __name__ == '__main__':
    melon_file = os.path.abspath('../data/xigua_3.0.csv')
    water_melon = pd.read_csv(melon_file, index_col='编号')

    discrete_cols = ['色泽', '根蒂', '敲声', '纹理', '脐部', '触感']
    discrete_cols = list(zip(discrete_cols, [-99999]*len(discrete_cols)))

    continuous_cols = list(zip(continuous_cols_name,
        [-99999]*len(continuous_cols_name)))

    all_cols = discrete_cols + continuous_cols

    tmp_cols = all_cols.copy()
    construct_tree(water_melon, tmp_cols, with_lr=True) 
