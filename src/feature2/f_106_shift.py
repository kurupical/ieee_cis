import pandas as pd
import numpy as np
import tqdm
import os
pd.set_option('display.max_columns', 100)

def _agg(df_train, df_test, agg_col, target_col, agg_type, compare_amt=False):

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

    w_df = pd.concat([df_train[[agg_col, target_col]], df_test[[agg_col, target_col]]]).reset_index(drop=True)
    w_df = w_df.groupby([agg_col])[target_col].agg([agg_type]).reset_index().rename(columns={agg_type: new_col_name})

    if agg_type in ["cumsum", "cumcount", "shift", "diff"]:
        df_train[new_col_name] = w_df[new_col_name].head(len(df_train)).reset_index(drop=True)
        df_test[new_col_name] = w_df[new_col_name].tail(len(df_test)).reset_index(drop=True)

        if agg_type == "cumsum":
            df_train["div_{}_cumcount".format(new_col_name)] = \
                df_train[new_col_name] / df_train["cumcount_{}".format(agg_col)]
            df_test["div_{}_cumcount".format(new_col_name)] = \
                df_test[new_col_name] / df_test["cumcount_{}".format(agg_col)]

        if compare_amt:
            df_train["diff_TransactionAmt_{}".format(new_col_name)] = \
                df_train["TransactionAmt"] - df_train[new_col_name]
            df_test["diff_TransactionAmt_{}".format(new_col_name)] = \
                df_test["TransactionAmt"] - df_test[new_col_name]
    else:
        w_df.index = list(w_df[agg_col])
        w_df = w_df[new_col_name].to_dict()

        df_train[new_col_name] = df_train[agg_col].map(w_df)
        df_test[new_col_name] = df_test[agg_col].map(w_df)

        if agg_type in ["mean", "std", "sum"]:
            df_train["div_{}".format(new_col_name)] = df_train[new_col_name] / df_train[target_col]
            df_test["div_{}".format(new_col_name)] = df_test[new_col_name] / df_test[target_col]

    return df_train, df_test

def id_aggregates(df_train, df_test, agg_cols, target_cols, agg_types, compare_amt=False):
    for agg_col in tqdm.tqdm(agg_cols):
        for target_col in target_cols:
            for agg_type in agg_types:
                df_train, df_test = _agg(df_train, df_test, agg_col, target_col, agg_type, compare_amt)
    return df_train, df_test

def main():
    df_train = pd.read_feather("../../data/baseline/train/baseline.feather")
    df_test = pd.read_feather("../../data/baseline/test/baseline.feather")

    original_features = df_train.columns

    agg_cols = ["TEMP__uid2+DT2"]
    df_train, df_test = id_aggregates(df_train, df_test,
                                      agg_cols=agg_cols,
                                      target_cols=["TransactionDT"],
                                      agg_types=["diff"])
    df_train, df_test = id_aggregates(df_train, df_test,
                                      agg_cols=agg_cols,
                                      target_cols=["D1", "D10"],
                                      agg_types=["shift"])
    df_train, df_test = id_aggregates(df_train, df_test,
                                      agg_cols=agg_cols,
                                      target_cols=["C{}".format(x) for x in range(1, 14+1)],
                                      agg_types=["diff"])
    v126_137 = ["V{}".format(x) for x in range(126, 137+1)]
    v202_216 = ["V{}".format(x) for x in range(202, 216+1)]
    v263_278 = ["V{}".format(x) for x in range(263, 278+1)]
    v306_321 = ["V{}".format(x) for x in range(306, 321+1)]
    df_train, df_test = id_aggregates(df_train, df_test,
                                      agg_cols=agg_cols,
                                      target_cols=v126_137+v202_216+v263_278+v306_321,
                                      agg_types=["diff"],
                                      compare_amt=True)


    df_train = df_train[[x for x in df_train.columns if x not in original_features]]
    df_test = df_test[[x for x in df_test.columns if x not in original_features]]

    print("train shape: {}".format(df_train.shape))
    print(df_train.head(5))
    print(df_train.describe())
    print("test shape: {}".format(df_test.shape))
    print(df_test.head(5))
    print(df_test.describe())

    df_train.reset_index(drop=True).to_feather("../../data/106_shift/train/shift.feather")
    df_test.reset_index(drop=True).to_feather("../../data/106_shift/test/shift.feather")

if __name__ == "__main__":
    print(os.path.basename(__file__))
    main()
