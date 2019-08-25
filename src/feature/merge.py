import pandas as pd
import numpy as np
import gc
import glob
from tqdm import tqdm
import time

def valid_card(df_train, df_test):
    for col in ["card1"]:
        valid_card = df_train[col].value_counts()
        valid_card = valid_card[valid_card > 10]
        valid_card = list(valid_card.index)

        df_train[col] = np.where(df_train[col].isin(valid_card), df_train[col], np.nan)
        df_test[col] = np.where(df_test[col].isin(valid_card), df_test[col], np.nan)

    return df_train, df_test

def drop_minority_card123(df_train, df_test):
    print("original length: {}".format(len(df_train)))

    # trainにしかないcard123はノイズになるので消す
    df_train_agg = df_train.groupby(["card1", "card2", "card3"])["TransactionID"].count()
    df_test_agg = df_test.groupby(["card1", "card2", "card3"])["TransactionID"].count()

    df_all = pd.concat([df_train_agg, df_test_agg], axis=1).fillna(0)
    df_all.columns = ["train", "test"]
    df_all["traintest_ratio"] = df_all["train"] / df_all["test"]

    df_all = df_all.query("traintest_ratio < 5").reset_index()
    df_train = pd.merge(df_train, df_all, how="inner", on=["card1", "card2", "card3"])
    df_train = df_train.drop(["train", "test", "traintest_ratio"], axis=1)

    print("after postprocessing length: {}".format(len(df_train)))
    return df_train, df_test

# time.sleep(60*60*2)
merge_features = []
# merge_features = glob.glob("../../data/date/train/*.feather")
merge_features.extend(glob.glob("../../data/transactionDT/train/*.feather"))
merge_features.extend(glob.glob("../../data/countencoding/train/*.feather"))
merge_features.extend(glob.glob("../../data/decimal/train/*.feather"))
merge_features.extend(glob.glob("../../data/identify/train/*.feather"))
merge_features.extend(glob.glob("../../data/basic_feature/train/*.feather"))

# merge_features.extend(glob.glob("../../data/agg/card2/train/*"))
merge_features.extend(glob.glob("../../output/aggfeat_top640_20190819220508_extract/feathers/train_*.feather"))

merge_features = [x.replace("train", "{}") for x in merge_features]
df_train = pd.read_feather("../../data/original/train_all.feather")
# print(f)
dfs = [df_train]
dfs.extend([pd.read_feather(x.format("train")) for x in merge_features])
df_train = pd.concat(dfs, axis=1)

print("test")
df_test = pd.read_feather("../../data/original/test_all.feather")
dfs = [df_test]
dfs.extend([pd.read_feather(x.format("test")) for x in merge_features])
df_test = pd.concat(dfs, axis=1)

print(df_train)
# df_train, df_test = drop_minority_card123(df_train, df_test)
# df_train, df_test = valid_card(df_train, df_test)

drop_cols = []
df_train = df_train.drop(drop_cols, axis=1)
df_test = df_test.drop(drop_cols, axis=1)
df_train.to_feather("../../data/merge/train_merge.feather")
df_test.to_feather("../../data/merge/test_merge.feather")
