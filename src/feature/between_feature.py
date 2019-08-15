import pandas as pd
import numpy as np
from itertools import combinations

def diff_2feature(df_train, df_test, col1, col2):
    return (df_train[col1] - df_train[col2]).astype(np.float32), (df_test[col1] - df_test[col2]).astype(np.float32)

def div_2feature(df_train, df_test, col1, col2):
    if (df_train[col1] == 0).sum() > (df_train[col2] == 0).sum():
        ret_train = df_train[col1] / (df_train[col2] - 1e-5)
        ret_test = df_test[col1] / (df_test[col2] - 1e-5)
    else:
        ret_train = df_train[col2] / (df_train[col1] - 1e-5)
        ret_test = df_test[col2] / (df_test[col1] - 1e-5)

    return ret_train.astype(np.float32), ret_test.astype(np.float32)

def make_list_2features(func, cols):
    ret_list = []
    for col in combinations(cols, 2):
        col1 = col[0]
        col2 = col[1]
        col_name = "{}_{}_{}".format(func.__name__, col1, col2)
        ret_list.append({"func": func, "col_name": col_name, "params": {"col1": col1, "col2": col2}})

    return ret_list
