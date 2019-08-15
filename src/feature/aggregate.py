import pandas as pd
import numpy as np
import gc
from tqdm import tqdm
import os
import time

from src.feature.common import reduce_mem_usage

class FeatureAggregateDivDiff:
    def __init__(self, df_train, df_test, is_shuffle=True):
        self.df_train = df_train
        self.df_test = df_test

        target_cols = ["card1", "card2", "card3", "card4", "card5", "card6", "ProductCD"]
        target_cols.extend(["M{}".format(x) for x in np.arange(1, 9 + 1)])
        target_cols.extend(["id_{}".format(x) for x in np.arange(12, 38 + 1)])

        self.agg_cols = ["C{}".format(x) for x in range(1, 14)]
        self.agg_cols.extend(["V{}".format(x) for x in range(1, 339 + 1)])
        self.agg_cols.extend(["D{}".format(x) for x in range(1, 15 + 1)])
        self.agg_cols.extend(["id_{0:02d}".format(x) for x in range(1, 11 + 1)])
        self.agg_cols.extend(["TransactionAmt"])

        aggregates = ["mean", "std"]
        # operators = [self.agg_div, self.agg_diff]
        operators = [self.agg_div]

        self.df_train[target_cols] = self.df_train[target_cols].fillna("nan")
        self.df_test[target_cols] = self.df_test[target_cols].fillna("nan")

        self.feature_engineering_list = []
        for tcol in target_cols:
            for acol in self.agg_cols:
                for agg in aggregates:
                    for operator in operators:
                        self.feature_engineering_list.append({
                            "func": operator,
                            "col_name": "{}_{}_{}_{}".format(operator.__name__, agg, acol, tcol),
                            "params": {
                                "agg_func": agg,
                                "agg_col": acol,
                                "target_col": tcol
                            }
                        })

        if is_shuffle:
            np.random.shuffle(self.feature_engineering_list)

        # aggregate
        self.agg_df_dict = {}
        for agg in tqdm(aggregates, desc="aggregate function"):
            self.agg_df_dict[agg] = {}
            for tcol in tqdm(target_cols, desc="aggregate columns"):
                if agg == "mean":
                    self.agg_df_dict[agg][tcol] = df_train.append(df_test)[self.agg_cols + [tcol]].groupby(tcol).mean()
                if agg == "std":
                    self.agg_df_dict[agg][tcol] = df_train.append(df_test)[self.agg_cols + [tcol]].groupby(tcol).std()
                if agg == "max":
                    self.agg_df_dict[agg][tcol] = df_train.append(df_test)[self.agg_cols + [tcol]].groupby(tcol).max()
                if agg == "min":
                    self.agg_df_dict[agg][tcol] = df_train.append(df_test)[self.agg_cols + [tcol]].groupby(tcol).max()

                self.agg_df_dict[agg][tcol].columns = self.agg_cols
                self.agg_df_dict[agg][tcol] = self.agg_df_dict[agg][tcol].reset_index()

    def agg_div(self, df_train, df_test, agg_func, target_col, agg_col):
        # あとで df_train, df_testを引数から消す

        # aggregation
        df_merge = pd.merge(self.df_train.append(self.df_test)[[target_col]], self.agg_df_dict[agg_func][target_col], how="left")
        df_train_agg = df_merge[agg_col].iloc[:len(self.df_train)].values.reshape(-1)
        df_test_agg = df_merge[agg_col].iloc[len(self.df_train):].values.reshape(-1)

        ret_df_train = self.df_train[agg_col].values / (df_train_agg - 1e-5)
        ret_df_test = self.df_test[agg_col].values / (df_test_agg - 1e-5)

        return ret_df_train.astype(np.float32), ret_df_test.astype(np.float32)

    def agg_diff(self, df_train, df_test, agg_func, target_col, agg_col):
        # あとで df_train, df_testを引数から消す

        # aggregation
        df_merge = pd.merge(self.df_train.append(self.df_test)[[target_col]], self.agg_df_dict[agg_func][target_col], how="left")
        df_train_agg = df_merge[agg_col].iloc[:len(self.df_train)].values.reshape(-1)
        df_test_agg = df_merge[agg_col].iloc[len(self.df_train):].values.reshape(-1)

        ret_df_train = self.df_train[agg_col].values - df_train_agg
        ret_df_test = self.df_test[agg_col].values - df_test_agg

        return ret_df_train.astype(np.float32), ret_df_test.astype(np.float32)

def aggregate(input_path_train, input_path_test, output_dir):
    df_train = pd.read_feather(input_path_train)
    df_test = pd.read_feather(input_path_test)

    target_cols = ["card1", "card2", "card3", "card4", "card5", "card6", "ProductCD"]
    target_cols.extend(["M{}".format(x) for x in np.arange(1, 9 + 1)])
    target_cols.extend(["id_{}".format(x) for x in np.arange(12, 38 + 1)])

    agg_cols = ["C{}".format(x) for x in range(1, 14)]
    agg_cols.extend(["V{}".format(x) for x in range(1, 339 + 1)])
    agg_cols.extend(["D{}".format(x) for x in range(1, 15 + 1)])
    agg_cols.extend(["id_{0:02d}".format(x) for x in range(1, 11 + 1)])
    agg_cols.extend(["TransactionAmt"])
    df_train[target_cols] = df_train[target_cols].fillna("nan")
    df_test[target_cols] = df_test[target_cols].fillna("nan")

    for t_col in target_cols:
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
            df_result[col] = (df_train[a_col].values / df_agg_mean[a_col].iloc[:len(df_train)].values).astype(
                np.float32)
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

if __name__ == "__main__":
    aggregate(input_path_train="../../data/original/train_all.feather",
              input_path_test="../../data/original/test_all.feather",
              output_dir="../../data/agg")