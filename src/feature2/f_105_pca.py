import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def apply_PCA(df_train, df_test, target_cols, n_components):
    w_ary = pd.concat([df_train[target_cols], df_test[target_cols]]).fillna(0).values

    w_ary = StandardScaler().fit_transform(w_ary)
    pca_ary = PCA(n_components=n_components).fit_transform(w_ary)

    col_names = ["PCA_{}_{}".format(target_cols[0], x) for x in range(n_components)]

    for i in range(len(col_names)):
        df_train[col_names[i]] = pca_ary[:len(df_train), i]
        df_test[col_names[i]] = pca_ary[len(df_train):, i]

    return df_train, df_test

def main():
    df_train = pd.read_feather("../../data/baseline/train/baseline.feather")
    df_test = pd.read_feather("../../data/baseline/test/baseline.feather")

    original_features = df_train.columns

    df_train, df_test = apply_PCA(df_train, df_test,
                                  target_cols=["D{}".format(x) for x in range(1, 15 + 1)],
                                  n_components=3)
    df_train, df_test = apply_PCA(df_train, df_test,
                                  target_cols=["C{}".format(x) for x in range(1, 14 + 1)],
                                  n_components=3)

    groups = []
    groups.append(["V{}".format(x) for x in range(1, 11+1)])
    groups.append(["V{}".format(x) for x in range(12, 34+1)])
    groups.append(["V{}".format(x) for x in range(35, 52+1)])
    groups.append(["V{}".format(x) for x in range(53, 74+1)])
    groups.append(["V{}".format(x) for x in range(75, 94+1)])
    groups.append(["V{}".format(x) for x in range(95, 137+1)])
    groups.append(["V{}".format(x) for x in range(138, 166+1)])
    groups.append(["V{}".format(x) for x in range(167, 216+1)])
    groups.append(["V{}".format(x) for x in range(217, 278+1)])
    groups.append(["V{}".format(x) for x in range(279, 321+1)])
    groups.append(["V{}".format(x) for x in range(322, 339+1)])

    df_train = df_train[[x for x in df_train.columns if x not in original_features]]
    df_test = df_test[[x for x in df_test.columns if x not in original_features]]

    df_train.to_feather("../../data/105_pca/train/pca.feather")
    df_test.to_feather("../../data/105_pca/test/pca.feather")

if __name__ == "__main__":
    main()