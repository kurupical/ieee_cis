import pandas as pd
import numpy as np
import sys
import tqdm

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

df_train = pd.read_feather("../../data/baseline/train/baseline.feather")
df_test = pd.read_feather("../../data/baseline/test/baseline.feather")

original_features = df_train.columns

"""
df_train = make_nan_pattern(df_train)
df_test = make_nan_pattern(df_test)
"""

df_train = df_train[[x for x in df_train.columns if x not in original_features]]
df_test = df_test[[x for x in df_test.columns if x not in original_features]]

print("train shape: {}".format(df_train.shape))
print(df_train.head(5))
print(df_train.describe())
print("test shape: {}".format(df_test.shape))
print(df_test.head(5))
print(df_test.describe())


df_train.to_feather("../../data/103_pattern/train/pattern.feather")
df_test.to_feather("../../data/103_pattern/test/pattern.feather")
