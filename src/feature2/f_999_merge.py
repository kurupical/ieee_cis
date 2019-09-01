import pandas as pd
import numpy as np
import gc
import glob
from tqdm import tqdm
import time


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

def main():
    merge_features = []
    merge_features = glob.glob("../../data/101_agg_id/train/*.feather")
    merge_features.extend(glob.glob("../../data/102_countencoding/train/*.feather"))
    merge_features.extend(glob.glob("../../data/103_pattern/train/*.feather"))
    merge_features.extend(glob.glob("../../data/104_pattern/train/*.feather"))
    merge_features.extend(glob.glob("../../data/104_1_agg_id/train/*.feather"))
    merge_features = [x.replace("train", "{}") for x in merge_features]

    df_train = pd.read_feather("../../data/baseline/train/baseline.feather")
    # print(f)
    dfs = [df_train]
    dfs.extend([pd.read_feather(x.format("train")) for x in merge_features])
    df_train = pd.concat(dfs, axis=1)
    df_train = postprocess(df_train)
    drop_cols = [x for x in df_train.columns if x[:6] in ["TEMP__"]]
    print("drop_cols: {}".format(drop_cols))
    drop_cols.extend(["TransactionDT", "id_30", "id_31", "id_33"])
    df_train = df_train.drop(drop_cols, axis=1)
    df_train.to_feather("../../data/merge/train_merge.feather")

    del df_train
    gc.collect()

    print("test")
    df_test = pd.read_feather("../../data/baseline/test/baseline.feather")
    dfs = [df_test]
    dfs.extend([pd.read_feather(x.format("test")) for x in merge_features])
    df_test = pd.concat(dfs, axis=1)
    df_test = postprocess(df_test)
    df_test = df_test.drop(drop_cols, axis=1)

    df_test.to_feather("../../data/merge/test_merge.feather")

if __name__ == "__main__":
    main()