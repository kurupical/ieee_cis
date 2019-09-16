import pandas as pd
import numpy as np
import gc
import glob
from tqdm import tqdm
import time
from src.feature.common import reduce_mem_usage

np.random.seed(0)

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

# print("waiting...")
# time.sleep(60*60*0.2)

def main(merge_features=None, nrows=None):
    if merge_features is None:
        merge_features = []
        merge_features = glob.glob("../../data/101_agg_id/train/*.feather")
        merge_features.extend(glob.glob("../../data/102_countencoding/train/*.feather"))
        merge_features.extend(glob.glob("../../data/103_pattern/train/*.feather"))
        merge_features.extend(glob.glob("../../data/104_pattern/train/*.feather"))
        merge_features.extend(glob.glob("../../data/104_1_agg_id/train/*.feather"))
        merge_features.extend(glob.glob("../../data/105_pca/train/*.feather"))
        merge_features.extend(glob.glob("../../data/101_agg_id_V/train/*.feather"))
        # merge_features.extend(glob.glob("../../data/106_shift/train/*.feather"))
        merge_features.extend(glob.glob("../../data/108_pattern/train/*.feather"))
        merge_features = [x.replace("train", "{}") for x in merge_features]

    train_idx = np.arange(590540)
    test_idx = np.arange(506691)
    df_train = pd.read_feather("../../data/baseline/train/baseline.feather")
    if nrows is not None:
        train_idx = np.sort(np.concatenate([df_train[df_train["isFraud"]==1].index.to_list(),
                                            np.random.choice(train_idx, nrows, replace=False)]))
        test_idx = np.sort(np.random.choice(test_idx, nrows, replace=False))
    # print(f)
    df_train = df_train.iloc[train_idx]
    dfs = [df_train]
    drop_cols = pd.read_csv("../feature2/cols.csv")["column"].values
    dfs.extend([pd.read_feather(x.format("train")).iloc[train_idx].drop(drop_cols, axis=1, errors="ignore") for x in merge_features])
    df_train = pd.concat(dfs, axis=1)
    df_train = postprocess(df_train)

    del dfs
    gc.collect()
    df_train = df_train.drop(drop_cols, axis=1, errors="ignore") #なぜか入れないとちゃんと削除されない。。
    drop_cols_2 = [x for x in df_train.columns if x[:6] in ["TEMP__"]]
    drop_cols_2.extend(["TransactionDT", "id_30", "id_31", "id_33"])
    print(df_train.shape)
    print("drop_cols: {}".format(drop_cols_2))
    df_train = df_train.drop(drop_cols_2, axis=1, errors="ignore")
    df_train.reset_index(drop=True).to_feather("../../data/merge/train_merge.feather")

    assert(len(df_train) == len(train_idx))
    print(df_train.shape)
    train_col_nums = len(df_train.columns)

    del df_train
    gc.collect()

    print("test")
    df_test = pd.read_feather("../../data/baseline/test/baseline.feather").iloc[test_idx]
    dfs = [df_test]
    dfs.extend([pd.read_feather(x.format("test")).iloc[test_idx].drop(drop_cols, axis=1, errors="ignore") for x in merge_features])
    df_test = pd.concat(dfs, axis=1)
    del dfs
    gc.collect()
    df_test = postprocess(df_test)
    df_test = df_test.drop(drop_cols, axis=1, errors="ignore") #なぜか入れないとちゃんと削除されない。。
    df_test = df_test.drop(drop_cols_2, axis=1, errors="ignore")

    print(df_test.shape)

    assert(len(df_test) == len(test_idx))
    test_col_nums = len(df_test.columns)

    assert(train_col_nums-1 == test_col_nums)
    df_test.reset_index(drop=True).to_feather("../../data/merge/test_merge.feather")

    print(df_test.isnull().sum().sort_values(ascending=False))

if __name__ == "__main__":
    main()