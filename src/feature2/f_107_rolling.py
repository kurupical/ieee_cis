import pandas as pd
import numpy as np
import tqdm
import os
pd.set_option('display.max_columns', 100)

def _agg(df_train, df_test, agg_col, target_col, agg_type, agg_time):

    """

    :param df_train:
    :param df_test:
    :param agg_col: groupbyする項目
    :param target_col: 集計される項目
    :param agg_type: 集計方法:mean, sum, cumsum, など
    :return:
    """

    class Rolling:
        def __init__(self, agg_type, agg_time, target_col):
            self.agg_time = agg_time
            self.agg_type = agg_type
            self.target_col = target_col

        def apply(self, x):
            print(x)
            print(x.rolling(self.agg_time).sum())
            if self.agg_type == "sum":
                return x.rolling(self.agg_time).sum()
            if self.agg_type == "count":
                return x.rolling(self.agg_time).count()
            if self.agg_type == "mean":
                return x.rolling(self.agg_time).mean()

    print("agg: {}, target: {}, agg_type: {}".format(agg_col, target_col, agg_type))
    new_col_name = "{}_groupby{}_rolling_{}_time_{}".format(target_col, agg_col, agg_type, agg_time)
    df_train.index = df_train["TEMP__DT"]
    df_test.index = df_test["TEMP__DT"]

    rolling = Rolling(agg_type, agg_time, target_col)

    df_train[new_col_name] = df_train[[agg_col, target_col]].groupby(target_col).transform(rolling.apply)
    df_test[new_col_name] = df_test[[agg_col, target_col]].groupby(target_col).transform(rolling.apply)

    return df_train, df_test

def id_aggregates(df_train, df_test, agg_cols, target_cols, agg_types, agg_times):
    for agg_col in tqdm.tqdm(agg_cols):
        for target_col in target_cols:
            for agg_type in agg_types:
                for agg_time in agg_times:
                    df_train, df_test = _agg(df_train, df_test, agg_col, target_col, agg_type, agg_time)
    return df_train, df_test

def main():
    df_train = pd.read_feather("../../data/baseline/train/baseline.feather")
    df_test = pd.read_feather("../../data/baseline/test/baseline.feather")

    original_features = df_train.columns

    agg_cols = ["card1", "card2", "card3", "card5",
                "TEMP__uid", "TEMP__uid2", "TEMP__uid3", "TEMP__uid4", "TEMP__uid5",
                "TEMP__uid2+DT", "TEMP__uid3+DT", "TEMP__uid4+DT", "TEMP__uid5+DT",
                "TEMP__uid2+DT+M4", "TEMP__uid3+DT+M4",
                "TEMP__uid2+DT2", "TEMP__uid3+DT2"]
    df_train, df_test = id_aggregates(df_train, df_test,
                                      agg_cols=agg_cols,
                                      target_cols=["TransactionAmt"],
                                      agg_types=["mean", "count", "sum"],
                                      agg_times=["15s", "1h", "1d", "7d"])

    df_train = df_train[[x for x in df_train.columns if x not in original_features]]
    df_test = df_test[[x for x in df_test.columns if x not in original_features]]

    df_train.reset_index(drop=True).to_feather("../../data/107_agg_id/train/rolling.feather")
    df_test.reset_index(drop=True).to_feather("../../data/107_agg_id/test/rolling.feather")

if __name__ == "__main__":
    print(os.path.basename(__file__))
    main()
