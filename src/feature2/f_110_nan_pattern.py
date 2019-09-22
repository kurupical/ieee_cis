
import pandas as pd
import numpy as np
import datetime
import sys
import os
from itertools import combinations
from src.feature.common import reduce_mem_usage

def make_nan_pattern(df):
    def join_string(x):
        return "".join(x.astype(str).values)

    print(sys._getframe().f_code.co_name)
    df["nan_pattern_id"] = df[["id_{0:02d}".format(x) for x in range(1, 38+1)]].isnull().astype(int).apply(join_string, axis=1)
    df["nan_pattern_C"] = df[["C{}".format(x) for x in range(1, 14+1)]].isnull().astype(int).apply(join_string, axis=1)
    df["nan_pattern_D"] = df[["D{}".format(x) for x in range(1, 15+1)]].isnull().astype(int).apply(join_string, axis=1)
    df["nan_pattern_M"] = df[["M{}".format(x) for x in range(1, 9+1)]].isnull().astype(int).apply(join_string, axis=1)
    df["nan_pattern_V"] = df[["V{}".format(x) for x in range(1, 339+1)]].isnull().astype(int).apply(join_string, axis=1)
    df["nan_pattern_M+M4"] = df["nan_pattern_M"].astype(str).str.cat(df["M4"].astype(str))
    return df

def main():
    df_train = pd.read_feather("../../data/original/train_all.feather")
    df_test = pd.read_feather("../../data/original/test_all.feather")

    original_features = df_train.columns

    df_train = make_nan_pattern(df_train)
    df_test = make_nan_pattern(df_test)

    df_train = df_train[[x for x in df_train.columns if x not in original_features]]
    df_test = df_test[[x for x in df_test.columns if x not in original_features]]

    print("train shape: {}".format(df_train.shape))
    print(df_train.head(5))
    print("test shape: {}".format(df_test.shape))
    print(df_test.tail(5))
    df_train.to_feather("../../data/110_nan_pattern/train/nan.feather")
    df_test.to_feather("../../data/110_nan_pattern/test/nan.feather")

if __name__ == "__main__":
    print(os.path.basename(__file__))
    main()