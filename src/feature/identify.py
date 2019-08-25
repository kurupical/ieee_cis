import numpy as np
import pandas as pd
import tqdm
pd.set_option('display.max_columns', 100)

def valid_card(df_train, df_test):
    for col in ["card1"]:
        valid_card = df_train[col].value_counts()
        valid_card = valid_card[valid_card > 10]
        valid_card = list(valid_card.index)

        df_train[col] = np.where(df_train[col].isin(valid_card), df_train[col], np.nan)
        df_test[col] = np.where(df_test[col].isin(valid_card), df_test[col], np.nan)

    return df_train, df_test

def identify():
    def to_feather(df, col_name, mode):
        if len(df) == df.isnull().sum():
            print("ERROR! ALL NULL: {}".format(col_name))
        pd.DataFrame(df.astype(np.float32).values, columns=[col_name]).reset_index(drop=True).to_feather("../../data/identify/{}/{}.feather".format(mode, col_name))
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
    df_train, df_test = valid_card(df_train, df_test)
    agg_cols = ['TransactionAmt']
    # agg_cols.extend(["id_{0:02d}".format(x) for x in range(1, 11+1)])
    # agg_cols.extend(["C{}".format(x) for x in range(1, 14+1)])

    for base_col in tqdm.tqdm(["card1", "card2", "W__uid1", "W__uid3"]):
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
            for div_date in tqdm.tqdm([1, 7, 30, 10000], desc="div_date"):
                # count encoding
                df_train["W__uid_date"] = df_train["W__uid_col"] + (df_train[date_col] // div_date).astype(str)
                df_test["W__uid_date"] = df_test["W__uid_col"] + (df_test[date_col] // div_date).astype(str)

                col_name = "agg_{}_{}_div{}_count".format(base_col[3:], date_col[3:], div_date)
                enc = df_train.groupby(["W__uid_date"]).agg({"TransactionID": "count"})["TransactionID"].to_dict()
                to_feather(df_train["W__uid_date"].map(enc), col_name=col_name, mode="train")
                enc = df_test.groupby(["W__uid_date"]).agg({"TransactionID": "count"})["TransactionID"].to_dict()
                to_feather(df_test["W__uid_date"].map(enc), col_name=col_name, mode="test")

                for agg_col in tqdm.tqdm(agg_cols, desc="agg_cols"):
                    agg_types = ["sum", "mean", "std", "cumsum"]
                    for agg_type in agg_types:
                        df_train[agg_col] = df_train[agg_col].astype(np.float32)
                        df_test[agg_col] = df_test[agg_col].astype(np.float32)
                        col_name = "agg_{}_{}_{}_{}".format(base_col[3:], date_col[3:], agg_col, agg_type)
                        enc = df_train.groupby(["W__uid_date"]).agg({agg_col: agg_type})[agg_col]

                        if agg_type == "cumsum":
                            to_feather(enc, col_name=col_name, mode="train")
                            to_feather(df_train[agg_col] / enc,
                                       col_name="div_{}".format(col_name), mode="train")
                        else:
                            to_feather(df_train["W__uid_date"].map(enc.to_dict()), col_name=col_name, mode="train")
                            to_feather(df_train[agg_col] / df_train["W__uid_date"].map(enc.to_dict()),
                                       col_name="div_{}".format(col_name), mode="train")
                        # to_feather(df_train[agg_col] - df_train["W__uid_date"].map(enc),
                        #            col_name="diff_{}".format(col_name), mode="train")
                        enc = pd.concat([df_train, df_test]).groupby(["W__uid_date"]).agg({agg_col: agg_type})[agg_col]
                        if agg_type == "cumsum":
                            to_feather(enc.tail(506691).reset_index(drop=True), col_name=col_name, mode="test")
                            to_feather(df_test[agg_col] / enc.tail(506691).values,
                                       col_name="div_{}".format(col_name), mode="test")
                        else:
                            to_feather(df_test["W__uid_date"].map(enc.to_dict()), col_name=col_name, mode="test")
                            to_feather(df_test[agg_col] / df_test["W__uid_date"].map(enc.to_dict()),
                                       col_name="div_{}".format(col_name), mode="test")
                        # to_feather(df_test[agg_col] - df_test["W__uid_date"].map(enc),
                        #        col_name="diff_{}".format(col_name), mode="test")
identify()