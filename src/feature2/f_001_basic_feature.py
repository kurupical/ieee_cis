
import pandas as pd
import numpy as np
import datetime
import sys
import os
from itertools import combinations
from src.feature.common import reduce_mem_usage

def make_nan_pattern(df):
    def join_string(x):
        return "".join(x.astype(str).values)

    print(sys._getframe().f_code.co_name)
    df["nan_pattern_id"] = df[["id_{0:02d}".format(x) for x in range(1, 38+1)]].isnull().astype(int).apply(join_string, axis=1)
    df["nan_pattern_C"] = df[["C{}".format(x) for x in range(1, 14+1)]].isnull().astype(int).apply(join_string, axis=1)
    df["nan_pattern_D"] = df[["D{}".format(x) for x in range(1, 15+1)]].isnull().astype(int).apply(join_string, axis=1)
    df["nan_pattern_M"] = df[["M{}".format(x) for x in range(1, 9+1)]].isnull().astype(int).apply(join_string, axis=1)
    df["nan_pattern_V"] = df[["V{}".format(x) for x in range(1, 339+1)]].isnull().astype(int).apply(join_string, axis=1)
    df["nan_pattern_M+M4"] = df["nan_pattern_M"].astype(str).str.cat(df["M4"].astype(str))
    return df

def datefeature(df):
    # https://www.kaggle.com/kyakovlev/ieee-gb-2-make-amount-useful-again
    print(sys._getframe().f_code.co_name)

    START_DATE = datetime.datetime.strptime('2017-11-30', '%Y-%m-%d')
    df['TEMP__DT'] = df['TransactionDT'].apply(lambda x: (START_DATE + datetime.timedelta(seconds=x)))
    df['TEMP__DT_M'] = (df['TEMP__DT'].dt.year - 2017) * 12 + df['TEMP__DT'].dt.month
    df['TEMP__DT_W'] = (df['TEMP__DT'].dt.year - 2017) * 52 + df['TEMP__DT'].dt.weekofyear
    df['TEMP__DT_D'] = (df['TEMP__DT'].dt.year - 2017) * 365 + df['TEMP__DT'].dt.dayofyear
    df['TEMP__DT_H'] = (df['TEMP__DT'].dt.year - 2017) * 365 + df['TEMP__DT'].dt.dayofyear * 24 + df["TEMP__DT"].dt.hour

    df["DT_year-half"] = (df["TEMP__DT"].dt.month % 6).astype(str) # original
    df["DT_year-quarter"] = (df["TEMP__DT"].dt.month % 3).astype(str) # original
    df['DT_hour'] = df['TEMP__DT'].dt.hour
    df['DT_day_week'] = df['TEMP__DT'].dt.dayofweek
    df['DT_day'] = df['TEMP__DT'].dt.day
    df["DT_isDecember"] = (df["TEMP__DT"].dt.month == 12).astype(int)

    # D9 column
    df['D9'] = np.where(df['D9'].isna(), 0, 1)
    return df

def remove_minor_cat(df_train, df_test, target_cols):

    # https://www.kaggle.com/kyakovlev/ieee-gb-2-make-amount-useful-again
    print(sys._getframe().f_code.co_name)
    for col in target_cols:
        valid_card = pd.concat([df_train[[col]], df_test[[col]]])
        valid_card = valid_card[col].value_counts()
        valid_card = valid_card[valid_card > 2]
        valid_card = list(valid_card.index)

        # df_train[col] = np.where(df_train[col].isin(df_test[col]), df_train[col], np.nan)
        # df_test[col] = np.where(df_test[col].isin(df_train[col]), df_test[col], np.nan)

        df_train[col] = np.where(df_train[col].isin(valid_card), df_train[col], np.nan)
        df_test[col] = np.where(df_test[col].isin(valid_card), df_test[col], np.nan)

    return df_train, df_test

def email_domain(df):
    # https://www.kaggle.com/kyakovlev/ieee-gb-2-make-amount-useful-again
    print(sys._getframe().f_code.co_name)

    p = 'P_emaildomain'
    r = 'R_emaildomain'
    uknown = 'email_not_provided'

    df[p] = df[p].fillna(uknown)
    df[r] = df[r].fillna(uknown)

    # Check if P_emaildomain matches R_emaildomain
    df['email_check'] = np.where((df[p] == df[r]) & (df[p] != uknown), 1, 0)

    df[p + '_prefix'] = df[p].apply(lambda x: x.split('.')[0])
    df[r + '_prefix'] = df[r].apply(lambda x: x.split('.')[0])
    return df

def device_info(df):
    # https://www.kaggle.com/kyakovlev/ieee-gb-2-make-amount-useful-again
    print(sys._getframe().f_code.co_name)

    ########################### Device info
    df['DeviceInfo'] = df['DeviceInfo'].fillna('unknown_device').str.lower()
    df['DeviceInfo_device'] = df['DeviceInfo'].apply(lambda x: ''.join([i for i in x if i.isalpha()]))
    df['DeviceInfo_version'] = df['DeviceInfo'].apply(lambda x: ''.join([i for i in x if i.isnumeric()]))

    ########################### Device info 2
    df['id_30'] = df['id_30'].fillna('unknown_device').str.lower()
    df['id_30_device'] = df['id_30'].apply(lambda x: ''.join([i for i in x if i.isalpha()]))
    df['id_30_version'] = df['id_30'].apply(lambda x: ''.join([i for i in x if i.isnumeric()]))

    ########################### Browser
    df['id_31'] = df['id_31'].fillna('unknown_device').str.lower()
    df['id_31_device'] = df['id_31'].apply(lambda x: ''.join([i for i in x if i.isalpha()]))

    return df

def amt_check(df_train, df_test):
    # https://www.kaggle.com/kyakovlev/ieee-gb-2-make-amount-useful-again
    print(sys._getframe().f_code.co_name)

    df_train['TransactionAmt_check'] = np.where(df_train['TransactionAmt'].isin(df_test['TransactionAmt']), 1, 0)
    df_test['TransactionAmt_check'] = np.where(df_test['TransactionAmt'].isin(df_train['TransactionAmt']), 1, 0)

    return df_train, df_test

def identify_id(df):
    # https://www.kaggle.com/kyakovlev/ieee-gb-2-make-amount-useful-again

    def fillna(x):
        if "nan" in x:
            return np.nan
        else:
            return x

    def d1_zero_to_d10(x):
        if x["D1"] == 0:
            return x.fillna(0)["D10"]
        else:
            return x["D1"]

    def nan_addr1_addr2(x):
        if np.isnan(x["addr1"]) and np.isnan(x["addr2"]):
            return np.nan
        else:
            return x["TEMP__uid3"]

    def d10_d12(x):
        if x["ProductCD"] == "C":
            return x["D12"]
        else:
            return x["D10"]

    print(sys._getframe().f_code.co_name)
    df["TEMP__D1_zero_to_D10"] = df[["D1", "D10"]].apply(d1_zero_to_d10, axis=1)
    df["TEMP__D10_D12"] = df[["D10", "D12", "ProductCD"]].apply(d10_d12, axis=1)

    df["TEMP__uid"] = df['card1'].astype(str)+'_'+df['card2'].astype(str)
    df["TEMP__uid2"] = df["TEMP__uid"].astype(str)+"_"+df['card3'].astype(str)+'_'+df['card5'].astype(str)+"_"# +df["P_emaildomain"].astype(str)
    df["TEMP__uid3"] = df["TEMP__uid2"].astype(str) + "_" + df['addr1'].fillna("none").astype(str) + '_' + df['addr2'].fillna("none").astype(str)
    # df["TEMP__uid3"] = df[["addr1", "addr2", "TEMP__uid3"]].apply(nan_addr1_addr2, axis=1).astype(str)

    df["TEMP__uid4"] = (df["V160"] - df["V159"]).astype(str)
    df["TEMP__uid5"] = df["addr2"].astype(str)+"_"+df["id_14"].astype(str)+"_"+df["id_19"].astype(str)+df["id_20"].astype(str)+df["P_emaildomain"].astype(str)
    df["TEMP__uid2+dist"] = df["TEMP__uid2"].astype(str) + "_" + df["dist1"].fillna("none").astype(str) + "_" + df["dist2"].fillna("none").astype(str)
    df["TEMP__uid2+DT"] = df["TEMP__uid2"].astype(str) + "_" + (df["TEMP__DT_D"] - df["D1"]).astype(str)
    df["TEMP__uid3+DT"] = df["TEMP__uid3"].astype(str) + "_" + (df["TEMP__DT_D"] - df["D1"]).astype(str)
    df["TEMP__uid2+dist+DT"] = df["TEMP__uid2+dist"].astype(str) + "_" + (df["TEMP__DT_D"] - df["D1"]).astype(str)
    # df["TEMP__uid2+DT"] = df["TEMP__uid2"].astype(str) + "_" + (df["TEMP__DT_D"] - df["TEMP__D1_zero_to_D10"]).astype(str)
    # df["TEMP__uid3+DT"] = df["TEMP__uid3"].astype(str) + "_" + (df["TEMP__DT_D"] - df["TEMP__D1_zero_to_D10"]).astype(str)
    df["TEMP__uid2+DT2"] = df["TEMP__uid2+DT"].astype(str)+"_"+(df["TEMP__DT_D"]-df["D10"]).fillna("none").astype(str)
    df["TEMP__uid3+DT2"] = df["TEMP__uid3+DT"].astype(str)+"_"+(df["TEMP__DT_D"]-df["D10"]).fillna("none").astype(str)
    df["TEMP__uid2+DT3"] = df["TEMP__uid2"].astype(str) + "_" + (df["TEMP__DT_D"] - df["TEMP__D10_D12"]).astype(str)
    df["TEMP__uid3+DT3"] = df["TEMP__uid3"].astype(str) + "_" + (df["TEMP__DT_D"] - df["TEMP__D10_D12"]).astype(str)
    df["TEMP__uid2+DT+D"] = df["TEMP__uid2"].astype(str)+"_"+(df["TEMP__DT_D"]-df["D1"]).astype(str)+"_"+df["TEMP__DT_D"].astype(str)
    df["TEMP__uid2+DT+W"] = df["TEMP__uid2"].astype(str)+"_"+(df["TEMP__DT_D"]-df["D1"]).astype(str)+"_"+df["TEMP__DT_W"].astype(str)
    df["TEMP__uid2+DT+H"] = df["TEMP__uid2"].astype(str)+"_"+(df["TEMP__DT_D"]-df["D1"]).astype(str)+"_"+df["TEMP__DT_H"].astype(str)
    df["TEMP__uid3+DT+D"] = df["TEMP__uid3"].astype(str)+"_"+(df["TEMP__DT_D"]-df["D1"]).astype(str)+"_"+df["TEMP__DT_D"].astype(str)
    df["TEMP__uid3+DT+W"] = df["TEMP__uid3"].astype(str)+"_"+(df["TEMP__DT_D"]-df["D1"]).astype(str)+"_"+df["TEMP__DT_W"].astype(str)
    df["TEMP__uid3+DT+H"] = df["TEMP__uid3"].astype(str)+"_"+(df["TEMP__DT_D"]-df["D1"]).astype(str)+"_"+df["TEMP__DT_H"].astype(str)
    df["TEMP__uid2+DT+M4"] = df["TEMP__uid2+DT"].astype(str)+"_"+(df["M4"]).astype(str)
    df["TEMP__uid3+DT+M4"] = df["TEMP__uid3+DT"].astype(str)+"_"+(df["M4"]).astype(str)

    for col in ["TEMP__uid", "TEMP__uid2", "TEMP__uid3", "TEMP__uid2+dist",
                "TEMP__uid2+DT", "TEMP__uid3+DT", "TEMP__uid2+dist+DT",
                "TEMP__uid2+DT2", "TEMP__uid3+DT2",
                "TEMP__uid2+DT3", "TEMP__uid3+DT3",
                "TEMP__uid2+DT+D", "TEMP__uid2+DT+W", "TEMP__uid2+DT+H",
                "TEMP__uid3+DT+D", "TEMP__uid3+DT+W", "TEMP__uid3+DT+H",
                "TEMP__uid2+DT+M4", "TEMP__uid3+DT+M4"]:
        df[col] = df[col].apply(fillna)

    # keyとして使う
    """
    df["uid2+DT"] = df["TEMP__uid2"].values
    df["uid3+DT"] = df["TEMP__uid3"].values
    df["uid2+DT+M4"] = df["TEMP__uid2+DT"].values
    df["uid3+DT+M4"] = df["TEMP__uid3+DT"].values
    """

    print("**DEBUG**")
    print(df["TEMP__uid2+DT"].value_counts())
    print(df["TEMP__uid3+DT"].value_counts())

    return df

def make_C_D_feature(df):
    print(sys._getframe().f_code.co_name)
    for d_col in ["D1", "D12"]:
        agg_cols = []
        agg_cols.extend(["C{}".format(x) for x in range(1, 14+1)])
        agg_cols.extend(["D5", "D11", "D13", "D14", "D15"])
        for agg_col in agg_cols:
            df["div_{}_{}".format(agg_col, d_col)] = df[agg_col] / df[d_col]

    for d_cols in combinations(["D1", "D2", "D3", "D4", "D5"], r=2):
        df["div_{}_{}".format(d_cols[0], d_cols[1])] = df[d_cols[0]] / df[d_cols[1]].replace(0, np.nan)
        df["diff_{}_{}".format(d_cols[0], d_cols[1])] = df[d_cols[0]] - df[d_cols[1]]

    return df

def fix_data_existing_only_test(df):
    print(sys._getframe().f_code.co_name)

    # C13がnanのデータ: C1-C15が0。異常データなのでnp.nanにするかな？
    idx_C13_is_nan = np.where(df["C13"].isnull())[0]
    for i in range(1, 14+1):
        df["C{}".format(i)].iloc[idx_C13_is_nan] = np.nan

    return df

def get_decimal(df):
    print(sys._getframe().f_code.co_name)
    df["AMT_Decimal"] = (df["TransactionAmt"] - df["TransactionAmt"].astype(int)).astype(str)
    df["AMT_Decimal_keta"] = [str(len(x)-2) for x in df["AMT_Decimal"].values]
    return df

def main():
    df_train = pd.read_feather("../../data/original/train_all.feather")
    df_test = pd.read_feather("../../data/original/test_all.feather")

    df_train = get_decimal(df_train)
    df_test = get_decimal(df_test)

    # df_train = reduce_mem_usage(df_train, mode="save")
    # df_test = reduce_mem_usage(df_test, mode="save")

    df_test = fix_data_existing_only_test(df_test)

    df_train = datefeature(df_train)
    df_test = datefeature(df_test)

    df_train = email_domain(df_train)
    df_test = email_domain(df_test)

    df_train = device_info(df_train)
    df_test = device_info(df_test)

    df_train = identify_id(df_train)
    df_test = identify_id(df_test)

    df_train = make_C_D_feature(df_train)
    df_test = make_C_D_feature(df_test)

    df_train, df_test = amt_check(df_train, df_test)

    df_train, df_test = remove_minor_cat(df_train, df_test,
                                         # target_cols=["card1", "TEMP__uid2+DT", "TEMP__uid3+DT", "TEMP__uid2+DT+M4", "TEMP__uid3+DT+M4"])
                                         target_cols=["card1"])

    print("train shape: {}".format(df_train.shape))
    print(df_train.head(5))
    print("test shape: {}".format(df_test.shape))
    print(df_test.tail(5))
    df_train.to_feather("../../data/baseline/train/baseline.feather")
    df_test.to_feather("../../data/baseline/test/baseline.feather")

if __name__ == "__main__":
    print(os.path.basename(__file__))
    main()