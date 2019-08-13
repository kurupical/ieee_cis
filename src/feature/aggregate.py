import pandas as pd
import numpy as np
import gc
from tqdm import tqdm
import os
import time

from src.feature.common import reduce_mem_usage

def agg(input_path_train, input_path_test, output_dir):

    df_train = pd.read_feather(input_path_train)
    df_test = pd.read_feather(input_path_test)

    target_cols = ["card1", "card2", "card3", "card4", "card5", "card6", "ProductCD"]
    target_cols.extend(["M{}".format(x) for x in np.arange(1, 9+1)])
    target_cols.extend(["id_{}".format(x) for x in np.arange(12, 38+1)])

    agg_cols = ["C{}".format(x) for x in range(1, 14)]
    agg_cols.extend(["V{}".format(x) for x in range(1, 339+1)])
    agg_cols.extend(["D{}".format(x) for x in range(1, 15+1)])
    agg_cols.extend(["id_{0:02d}".format(x) for x in range(1, 11+1)])
    agg_cols.extend(["TransactionAmt"])
    df_train[target_cols] = df_train[target_cols].fillna("nan")
    df_test[target_cols] = df_test[target_cols].fillna("nan")

    for t_col in target_cols[-5:]:
        print(t_col)
        df_agg_mean = df_train.append(df_test)[agg_cols + [t_col]].groupby(t_col).transform("mean")
        df_agg_mean.columns = agg_cols
        if not os.path.isdir("{}/train".format(output_dir)):
            os.makedirs("{}/train".format(output_dir))
        if not os.path.isdir("{}/test".format(output_dir)):
            os.makedirs("{}/test".format(output_dir))

        for a_col in tqdm(agg_cols, desc="train"):
            # mean
            df_result = pd.DataFrame()
            col = "agg_{}div{}mean".format(a_col, t_col)
            df_result[col] = (df_train[a_col].values / df_agg_mean[a_col].iloc[:len(df_train)].values).astype(np.float32)
            df_result.to_feather("{}/train/{}.feather".format(output_dir, col))

            """
            # std
            df_result = pd.DataFrame()
            col = "agg_{}diff{}std".format(a_col, t_col)
            df_result["agg_{}diff{}std".format(a_col, t_col)] = (df_train[a_col].values - df_agg_std[a_col].iloc[:len(df_train)]).astype(np.float32)
            df_result.to_feather("{}/train/{}.feather".format(output_dir, col))
            """
        for a_col in tqdm(agg_cols, desc="test"):
            # mean
            df_result = pd.DataFrame()
            col = "agg_{}div{}mean".format(a_col, t_col)
            df_result[col] = (df_test[a_col].values / df_agg_mean[a_col].iloc[len(df_train):].values).astype(np.float32)
            df_result.to_feather("{}/test/{}.feather".format(output_dir, col))

            """
            # std
            df_result = pd.DataFrame()
            col = "agg_{}diff{}std".format(a_col, t_col)
            df_result["agg_{}div{}std".format(a_col, t_col)] = (df_test[a_col].values - df_agg_std[a_col].iloc[len(df_train):].values).astype(np.float32)
            df_result.to_feather("{}/test/{}.feather".format(output_dir, col))
            """

# print("waiting...")
# time.sleep(60*60*1)

agg(input_path_train="../../data/original/train_all.feather",
    input_path_test="../../data/original/test_all.feather",
    output_dir="../../data/agg")