"""
基于对率回归进行划分选择的决策树算法，为表4.3生成决策树。

Author: Hao Wang
"""

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



