
import pandas as pd
import numpy as np
import datetime
import sys

def make_nan_pattern(df):
    def join_string(x):
        return "".join(x.astype(str).values)

    print(sys._getframe().f_code.co_name)
    df["nan_pattern_id"] = df[["id_{0:02d}".format(x) for x in range(1, 38+1)]].isnull().astype(int).apply(join_string, axis=1)
    df["nan_pattern_C"] = df[["C{}".format(x) for x in range(1, 14+1)]].isnull().astype(int).apply(join_string, axis=1)
    df["nan_pattern_D"] = df[["D{}".format(x) for x in range(1, 15+1)]].isnull().astype(int).apply(join_string, axis=1)
    df["nan_pattern_M"] = df[["M{}".format(x) for x in range(1, 9+1)]].isnull().astype(int).apply(join_string, axis=1)
    # df["nan_pattern_V"] = df[["V{}".format(x) for x in range(1, 339+1)]].isnull().astype(int).apply(join_string, axis=1)
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

        df_train[col] = np.where(df_train[col].isin(df_test[col]), df_train[col], np.nan)
        df_test[col] = np.where(df_test[col].isin(df_train[col]), df_test[col], np.nan)

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

    print(sys._getframe().f_code.co_name)
    df["TEMP__uid"] = df['card1'].astype(str)+'_'+df['card2'].astype(str)
    df["TEMP__uid2"] = df["TEMP__uid"].astype(str)+"_"+df['card3'].astype(str)+'_'+df['card5'].astype(str)
    df["TEMP__uid3"] = df["TEMP__uid2"].astype(str)+"_"+df['addr1'].astype(str)+'_'+df['addr2'].astype(str)
    df["TEMP__uid4"] = (df["V160"] - df["V159"]).astype(str)
    df["TEMP__uid2+DT"] = df["TEMP__uid2"].astype(str)+"_"+(df["TEMP__DT_D"]-df["D1"]).astype(str)
    df["TEMP__uid3+DT"] = df["TEMP__uid3"].astype(str)+"_"+(df["TEMP__DT_D"]-df["D1"]).astype(str)
    df["TEMP__uid2+DT+D"] = df["TEMP__uid2"].astype(str)+"_"+(df["TEMP__DT_D"]-df["D1"]).astype(str)+"_"+df["TEMP__DT_D"].astype(str)
    df["TEMP__uid2+DT+W"] = df["TEMP__uid2"].astype(str)+"_"+(df["TEMP__DT_D"]-df["D1"]).astype(str)+"_"+df["TEMP__DT_W"].astype(str)
    df["TEMP__uid2+DT+M4"] = df["TEMP__uid2+DT"].astype(str)+"_"+(df["M4"]).astype(str)
    df["TEMP__uid3+DT+M4"] = df["TEMP__uid3+DT"].astype(str)+"_"+(df["M4"]).astype(str)

    for col in ["TEMP__uid", "TEMP__uid2", "TEMP__uid3", "TEMP__uid4", "TEMP__uid2+DT", "TEMP__uid3+DT", "TEMP__uid2+DT+D", "TEMP__uid2+DT+W",
                "TEMP__uid2+DT+M4", "TEMP__uid3+DT+M4"]:
        df[col] = df[col].apply(fillna)

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

    return df

def fix_data_existing_only_test(df):
    print(sys._getframe().f_code.co_name)

    # C13がnanのデータ: C1-C15が0。異常データなのでnp.nanにするかな？
    idx_C13_is_nan = np.where(df["C13"].isnull())[0]
    for i in range(1, 14+1):
        df["C{}".format(i)].iloc[idx_C13_is_nan] = np.nan

    return df

df_train = pd.read_feather("../../data/original/train_all.feather")
df_test = pd.read_feather("../../data/original/test_all.feather")

df_train = make_nan_pattern(df_train)
df_test = make_nan_pattern(df_test)

df_test = fix_data_existing_only_test(df_test)

df_train = datefeature(df_train)
df_test = datefeature(df_test)

df_train, df_test = remove_minor_cat(df_train, df_test, target_cols=["card1"])

df_train = email_domain(df_train)
df_test = email_domain(df_test)

df_train = device_info(df_train)
df_test = device_info(df_test)

df_train = identify_id(df_train)
df_test = identify_id(df_test)

df_train = make_C_D_feature(df_train)
df_test = make_C_D_feature(df_test)

df_train, df_test = amt_check(df_train, df_test)

print("train shape: {}".format(df_train.shape))
print(df_train.head(5))
print("test shape: {}".format(df_test.shape))
print(df_test.tail(5))
df_train.to_feather("../../data/baseline/train/baseline.feather")
df_test.to_feather("../../data/baseline/test/baseline.feather")