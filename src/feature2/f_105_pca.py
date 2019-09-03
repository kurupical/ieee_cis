import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def apply_PCA(df_train, df_test, target_cols, n_components):
    w_ary = pd.concat([df_train[target_cols], df_test[target_cols]]).fillna(0).values

    w_ary = StandardScaler.fit_transform(w_ary)
    pca_ary = PCA(n_components=n_components).fit_transform(w_ary)

    col_names = ["PCA_{}_{}".format(target_cols[0], x) for x in range(n_components)]

    df_train[col_names] = pca_ary.head(len(df_train))
    df_test[col_names] = pca_ary.tail(len(df_test))

    return df_train, df_test


df_train = pd.read_feather("../../data/baseline/train/baseline.feather")
df_test = pd.read_feather("../../data/baseline/test/baseline.feather")

original_features = df_train.columns

df_train, df_test = apply_PCA(df_train, df_test,
                              target_cols=["D{}".format(x) for x in range(1, 15 + 1)])
df_train, df_test = apply_PCA(df_train, df_test,
                              target_cols=["C{}".format(x) for x in range(1, 14 + 1)])

df_train = df_train[[x for x in df_train.columns if x not in original_features]]
df_test = df_test[[x for x in df_test.columns if x not in original_features]]

print("train shape: {}".format(df_train.shape))
print(df_train.head(5))
print(df_train.describe())
print("test shape: {}".format(df_test.shape))
print(df_test.head(5))
print(df_test.describe())

df_train.to_feather("../../data/105_pca/train/pca.feather")
df_test.to_feather("../../data/105_pca/test/pca.feather")