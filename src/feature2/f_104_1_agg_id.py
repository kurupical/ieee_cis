import pandas as pd
import numpy as np
import tqdm
pd.set_option('display.max_columns', 100)
import os

def _agg(df_train, df_test, agg_col, target_col, agg_type):

    """

    :param df_train:
    :param df_test:
    :param agg_col: groupbyする項目
    :param target_col: 集計される項目
    :param agg_type: 集計方法:mean, sum, cumsum, など
    :return:
    """
    print("agg: {}, target: {}, agg_type: {}".format(agg_col, target_col, agg_type))
    new_col_name = "{}_groupby{}_{}".format(target_col, agg_col, agg_type)
    if agg_type == "cumcount":
        new_col_name = "cumcount_{}".format(agg_col)

    w_df = pd.concat([df_train[[agg_col, target_col]], df_test[[agg_col, target_col]]])
    w_df = w_df.groupby([agg_col])[target_col].agg([agg_type]).reset_index().rename(columns={agg_type: new_col_name})

    if agg_type in ["cumsum", "cumcount"]:
        df_train[new_col_name] = w_df[new_col_name].head(len(df_train)).reset_index(drop=True)
        df_test[new_col_name] = w_df[new_col_name].tail(len(df_test)).reset_index(drop=True)

        if agg_type == "cumsum":
            df_train["div_{}_cumcount".format(new_col_name)] = \
                df_train[new_col_name] / df_train["cumcount_{}".format(agg_col)]
            df_test["div_{}_cumcount".format(new_col_name)] = \
                df_test[new_col_name] / df_test["cumcount_{}".format(agg_col)]
    else:
        w_df.index = list(w_df[agg_col])
        w_df = w_df[new_col_name].to_dict()

        df_train[new_col_name] = df_train[agg_col].map(w_df)
        df_test[new_col_name] = df_test[agg_col].map(w_df)

        if agg_type in ["mean", "std", "sum"]:
            df_train["div_{}".format(new_col_name)] = df_train[new_col_name] / df_train[target_col]
            df_test["div_{}".format(new_col_name)] = df_test[new_col_name] / df_test[target_col]

    return df_train, df_test

def id_aggregates(df_train, df_test, agg_cols, target_cols, agg_types):
    for agg_col in tqdm.tqdm(agg_cols):
        for target_col in target_cols:
            for agg_type in agg_types:
                df_train, df_test = _agg(df_train, df_test, agg_col, target_col, agg_type)
    return df_train, df_test

def main():
    df_train = pd.read_feather("../../data/baseline/train/baseline.feather")
    df_train = pd.concat([df_train,
                          pd.read_feather("../../data/104_pattern/train/pattern.feather")], axis=1)
    df_test = pd.read_feather("../../data/baseline/test/baseline.feather")
    df_test = pd.concat([df_test,
                         pd.read_feather("../../data/104_pattern/test/pattern.feather")], axis=1)

    original_features = df_train.columns

    agg_cols = ["TEMP__uid4+DT",
                "TEMP__uid2+DT2", "TEMP__uid3+DT2",
                "TEMP__uid2+DT2+M4", "TEMP__uid3+DT2+M4"]
    target_cols = [x for x in df_train.columns if "div" not in x and "std" in x]
    df_train, df_test = id_aggregates(df_train, df_test,
                                      agg_cols=agg_cols,
                                      target_cols=target_cols,
                                      agg_types=["min", "max"])

    df_train = df_train[[x for x in df_train.columns if x not in original_features]]
    df_test = df_test[[x for x in df_test.columns if x not in original_features]]

    print("train shape: {}".format(df_train.shape))
    print(df_train.head(5))
    print(df_train.describe())
    print("test shape: {}".format(df_test.shape))
    print(df_test.head(5))
    print(df_test.describe())

    df_train.reset_index(drop=True).to_feather("../../data/104_1_agg_id/train/agg.feather")
    df_test.reset_index(drop=True).to_feather("../../data/104_1_agg_id/test/agg.feather")

if __name__ == "__main__":
    print(os.path.basename(__file__))
    main()
