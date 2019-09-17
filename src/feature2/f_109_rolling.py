import pandas as pd
import numpy as np
import tqdm
import os
import datetime as dt

def count_same_amt(df):
    def _search_same_amt(i):
        for j in range(1, i+1):
            df["count_same_amt_in_{}".format(i)] += (df["TransactionAmt"] == df["TransactionAmt"].shift(j)).astype(int)
        return df

    for i in [3, 10, 20]:
        df["count_same_amt_in_{}".format(i)] = 0
        df = _search_same_amt(i)

    return df

def transactionDT_diff(df):
    for i in [3, 10, 20, 100]:
        df["transactionDT_diff_{}".format(i)] = df["TransactionDT"] - df["TransactionDT"].shift(i)

    return df

def main():

    # 元データのロード
    df_train = pd.read_feather("../../data/baseline/train/baseline.feather")
    df_test = pd.read_feather("../../data/baseline/test/baseline.feather")

    original_features = df_train.columns

    df_train = count_same_amt(df_train)
    df_test = count_same_amt(df_test)

    df_train = transactionDT_diff(df_train)
    df_test = transactionDT_diff(df_test)

    df_train = df_train[[x for x in df_train.columns if x not in original_features]]
    df_test = df_test[[x for x in df_test.columns if x not in original_features]]

    # print(df_train.head(5))
    # print(df_test.tail(5))
    df_train.reset_index(drop=True).to_feather("../../data/109_rolling/train/rolling.feather")
    df_test.reset_index(drop=True).to_feather("../../data/109_rolling/test/rolling.feather")

if __name__ == "__main__":
    print(os.path.basename(__file__))
    main()
