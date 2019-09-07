import pandas as pd
import numpy as np
import gc
import glob
from tqdm import tqdm
from sklearn.preprocessing import OneHotEncoder
from sklearn.decomposition import SparsePCA
import time


def _get_categorical_features(df):
    numerics = ['int8', 'int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    feats = [col for col in list(df.columns) if df[col].dtype not in numerics]

    cat_cols = []
    cat_cols.extend(["ProductCD"])
    cat_cols.extend(["card{}".format(x) for x in np.arange(1, 6+1)])
    cat_cols.extend(["addr1", "addr2"])
    cat_cols.extend(["P_emaildomain", "R_emaildomain"])
    cat_cols.extend(["M{}".format(x) for x in np.arange(1, 9+1)])
    cat_cols.extend(["DeviceType", "DeviceInfo"])
    cat_cols.extend(["id_{}".format(x) for x in np.arange(12, 38+1)])

    cat_cols = [x for x in df.columns if x in cat_cols]
    feats.extend([x for x in cat_cols if x not in feats])
    return feats

def sparce_pca(df_train, df_test, n_components=30):

    print("sparce_pca")
    cols = ["card1", "card2", "card3", "card4", "card5", "card6", "addr1", "addr2", "id_19", "DeviceInfo"]
    for col in cols:
        valid = pd.concat([df_train[[col]], df_test[[col]]])
        valid = valid[col].value_counts()
        valid = valid[valid > 75]
        valid = list(valid.index)

        df_train[col] = np.where(df_train[col].isin(valid), df_train[col], "others")
        df_test[col] = np.where(df_test[col].isin(valid), df_test[col], "others")

    X_all = df_train.append(df_test)[cols]
    X_all = pd.get_dummies(X_all, columns=cols, sparse=True).astype(np.int8)
    print(X_all.shape)
    X_all = SparsePCA(n_components=n_components).fit_transform(X_all)

    col_names = ["cat_SpacePCA_{}".format(x) for x in range(n_components)]
    df_train = pd.DataFrame(X_all[:len(df_train)], columns=col_names)
    df_test = pd.DataFrame(X_all[len(df_train):], columns=col_names)

    return df_train, df_test


def postprocess(df):
    def fillna(df):
        cat_cols = []
        cat_cols.extend(["ProductCD"])
        cat_cols.extend(["addr1", "addr2"])
        cat_cols.extend(["P_emaildomain", "R_emaildomain"])
        cat_cols.extend(["M{}".format(x) for x in np.arange(1, 9 + 1)])
        cat_cols.extend(["DeviceType", "DeviceInfo"])
        cat_cols.extend(["id_{}".format(x) for x in np.arange(12, 38 + 1)])

        for col in cat_cols:
            df[col] = df[col].fillna("NAN_values").astype(str)
        return df

    df = fillna(df)

    return df

def main():
    def load_data(mode, merge_features, drop_cols):
        df = pd.read_feather("../../data/baseline/{}/baseline.feather".format(mode))
        dfs = [df]
        dfs.extend([pd.read_feather(x.format(mode)).drop(drop_cols, axis=1, errors="ignore") for x in merge_features])
        df = pd.concat(dfs, axis=1)

        return df

    merge_features = []
    merge_features = glob.glob("../../data/101_agg_id/train/*.feather")
    merge_features.extend(glob.glob("../../data/102_countencoding/train/*.feather"))
    merge_features.extend(glob.glob("../../data/103_pattern/train/*.feather"))
    merge_features.extend(glob.glob("../../data/104_pattern/train/*.feather"))
    merge_features.extend(glob.glob("../../data/104_1_agg_id/train/*.feather"))
    merge_features.extend(glob.glob("../../data/105_pca/train/*.feather"))
    merge_features = [x.replace("train", "{}") for x in merge_features]

    # 1.cat_featuresをSparcePCAする.
    """
    # メモリ対策のため、一回ここで特徴作る。
    df_train = load_data("train", merge_features)
    drop_cols = [x for x in df_train.columns if x[:6] in ["TEMP__"]]
    drop_cols.extend(["TransactionDT", "id_30", "id_31", "id_33"])
    df_train = df_train.drop(drop_cols, axis=1)
    cat_feats = _get_categorical_features(df_train.drop("TransactionID", axis=1))
    df_train = df_train[cat_feats]

    print("test")
    df_test = load_data("test", merge_features)
    df_test = df_test.drop(drop_cols, axis=1)
    df_test = df_test[cat_feats]

    df_train_pca, df_test_pca = sparce_pca(df_train, df_test)

    df_train_pca.to_feather("../../data/989_merge/train_pca.feather")
    df_test_pca.to_feather("../../data/989_merge/test_pca.feather")
    """

    # 2.連続値の特徴量 + pcaデータを結合
    drop_cols = []
    drop_cols.extend(["TransactionDT", "id_30", "id_31", "id_33"])
    drop_cols.extend(pd.read_csv("cols_nn.csv")["column"].values)
    df_train = load_data("train", merge_features, drop_cols)
    drop_cols_1 = [x for x in df_train.columns if x[:6] in ["TEMP__"]]
    df_train = df_train.drop(drop_cols_1, axis=1)
    cat_feats = _get_categorical_features(df_train.drop("TransactionID", axis=1))
    df_train = df_train.drop(cat_feats, axis=1)
    df_train_pca = pd.read_feather("../../data/989_merge/train_pca.feather")
    df_train = pd.concat([df_train, df_train_pca], axis=1)
    assert(len(df_train) == 590540)

    train_col_nums = len(df_train.columns)
    df_train.to_feather("../../data/989_merge/train_merge.feather")
    del df_train, df_train_pca
    gc.collect()

    df_test = load_data("test", merge_features, drop_cols).drop(cat_feats, axis=1)
    df_test = df_test.drop(drop_cols_1, axis=1)
    df_test_pca = pd.read_feather("../../data/989_merge/test_pca.feather")
    df_test = pd.concat([df_test, df_test_pca], axis=1)

    assert(len(df_test) == 506691)
    test_col_nums = len(df_test.columns)
    df_test.to_feather("../../data/989_merge/test_merge.feather")

    assert(train_col_nums-1 == test_col_nums)




if __name__ == "__main__":
    main()