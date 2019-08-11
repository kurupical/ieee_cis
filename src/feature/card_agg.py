import pandas as pd
import numpy as np
import gc
from tqdm import tqdm

from src.feature.common import reduce_mem_usage

def agg(input_path_train, input_path_test, output_dir):

    df_train = pd.read_feather(input_path_train)
    df_test = pd.read_feather(input_path_test)

    target_cols = ["card1", "card2"]
    agg_cols = ["C{}".format(x) for x in range(1, 14)]
    agg_cols.extend(["V{}".format(x) for x in range(1, 339+1)])
    agg_cols.extend(["D{}".format(x) for x in range(1, 15+1)])
    agg_cols.extend(["TransactionAmt"])

    for t_col in target_cols:
        df_agg_mean = df_train[agg_cols + [t_col]].groupby(t_col).transform("mean")
        df_agg_mean.columns = agg_cols
        df_agg_std = df_train[agg_cols + [t_col]].groupby(t_col).transform("std")
        df_agg_std.columns = agg_cols

        for a_col in tqdm(agg_cols, desc="train"):
            df_result = pd.DataFrame()
            df_result["TransactionID"] = df_train["TransactionID"]
            df_result["agg_{}div{}mean".format(a_col, t_col)] = (df_train[a_col] / df_agg_mean[a_col]).astype(np.float32)
            df_result["agg_{}div{}std".format(a_col, t_col)] = (df_train[a_col] / df_agg_std[a_col]).astype(np.float32)
            df_result.to_feather("{}/{}/train/{}.feather".format(output_dir, t_col, a_col))

        for a_col in tqdm(agg_cols, desc="test"):
            df_result = pd.DataFrame()
            df_result["TransactionID"] = df_test["TransactionID"]
            df_result["agg_{}div{}mean".format(a_col, t_col)] = (df_test[a_col] / df_agg_mean[a_col]).astype(np.float32)
            df_result["agg_{}div{}std".format(a_col, t_col)] = (df_test[a_col] / df_agg_std[a_col]).astype(np.float32)

            df_result.to_feather("{}/{}/test/{}.feather".format(output_dir, t_col, a_col))

agg(input_path_train="../../data/original/train_transaction.feather",
    input_path_test="../../data/original/test_transaction.feather",
    output_dir="../../data/agg")