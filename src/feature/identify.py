import numpy as np
import pandas as pd
import tqdm
pd.set_option('display.max_columns', 100)


def identify():
    def to_feather(df, col_name, mode):
        pd.DataFrame(df.values, columns=[col_name]).reset_index(drop=True).to_feather("../../data/identify/{}/{}.feather".format(mode, col_name))
    df_all = pd.concat([pd.read_feather("../../data/original/train_all.feather"),
                        pd.read_feather("../../data/original/test_all.feather")])

    df_result = pd.DataFrame()
    df_all["W__TransactionDay"] = (df_all["TransactionDT"] / 60 / 24 / 24).astype(int)
    df_all["W__diff_D1"] = (df_all["W__TransactionDay"] - df_all["D1"])
    # df_all["W__diff_D11"] = (df_all["W__TransactionDay"] - df_all["D11"])
    # df_all["W__diff_D12"] = (df_all["W__TransactionDay"] - df_all["D12"])
    # df_all["W__diff_D15"] = (df_all["W__TransactionDay"] - df_all["D15"])
    df_all["W__uid1"] = df_all["card1"].astype(str) + \
                            df_all["card2"].astype(str) + \
                            df_all["card3"].astype(str)
    df_all["W__uid2"] = df_all["W__uid1"].astype(str) + \
                             df_all["card5"].astype(str)
    df_all["W__uid3"] = df_all["W__uid2"].astype(str) + \
                             df_all["addr1"].astype(str) + \
                             df_all["addr2"].astype(str)

    df_train = df_all.head(590540).reset_index(drop=True)
    df_test = df_all.tail(506691).reset_index(drop=True)
    agg_cols = ['TransactionAmt']
    for base_col in tqdm.tqdm(["W__uid1", "W__uid2", "W__uid3"]):
        for date_col in ["W__diff_D1"]:
            df_train["W__uid_col"] = df_train[base_col].astype(str) + df_train[date_col].astype(str)
            df_test["W__uid_col"] = df_test[base_col].astype(str) + df_test[date_col].astype(str)

            # DT経過日
            col_name = "agg_{}_{}_{}".format(base_col[3:], date_col[3:], "diff_DTmin")
            enc = pd.concat([df_train, df_test]).groupby("W__uid_col")["TransactionDT"].min().to_dict()
            df_train["W__DTmin"] = df_train["W__uid_col"].map(enc)
            df_test["W__DTmin"] = df_test["W__uid_col"].map(enc)
            to_feather(df_train["TransactionDT"] - df_train["W__DTmin"], col_name=col_name, mode="train")
            to_feather(df_test["TransactionDT"] - df_test["W__DTmin"], col_name=col_name, mode="test")

            # Aggregate
            for div_date in [1, 7, 30, 10000]:
                # count encoding
                df_train["W__uid_date"] = df_train["W__uid_col"] + (df_train[date_col] // div_date).astype(str)
                df_test["W__uid_date"] = df_test["W__uid_col"] + (df_test[date_col] // div_date).astype(str)

                col_name = "agg_{}_{}_div{}_count".format(base_col[3:], date_col[3:], div_date)
                enc = df_train.groupby(["W__uid_date"]).agg({"TransactionID": "count"})["TransactionID"].to_dict()
                to_feather(df_train["W__uid_date"].map(enc), col_name=col_name, mode="train")
                enc = df_test.groupby(["W__uid_date"]).agg({"TransactionID": "count"})["TransactionID"].to_dict()
                to_feather(df_test["W__uid_date"].map(enc), col_name=col_name, mode="test")

                for agg_col in agg_cols:
                    for agg_type in ["sum", "mean", "std"]:
                        col_name = "agg_{}_{}_{}_{}_div{}".format(base_col[3:], date_col[3:], agg_col, agg_type, div_date)
                        enc = df_train.groupby(["W__uid_date"]).agg({agg_col: agg_type})[agg_col].to_dict()

                        to_feather(df_train["W__uid_date"].map(enc), col_name=col_name, mode="train")
                        to_feather(df_train[agg_col] / df_train["W__uid_date"].map(enc),
                                   col_name="div_{}".format(col_name), mode="train")
                        to_feather(df_train[agg_col] - df_train["W__uid_date"].map(enc),
                                   col_name="diff_{}".format(col_name), mode="train")
                        enc = pd.concat([df_train, df_test]).groupby(["W__uid_date"]).agg({agg_col: agg_type})[agg_col].to_dict()
                        to_feather(df_test["W__uid_date"].map(enc), col_name=col_name, mode="test")
                        to_feather(df_test[agg_col] / df_test["W__uid_date"].map(enc),
                                   col_name="div_{}".format(col_name), mode="test")
                        to_feather(df_test[agg_col] - df_test["W__uid_date"].map(enc),
                               col_name="diff_{}".format(col_name), mode="test")
identify()