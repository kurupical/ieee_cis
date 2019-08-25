import pandas as pd
import numpy as np

def to_feather(df, col_name, mode):
    pd.DataFrame(df.values, columns=[col_name]).reset_index(drop=True).to_feather(
        "../../data/basic_feature/{}/{}.feather".format(mode, col_name))

def make_identity_feature(df, output_dir):

    df_result = pd.DataFrame()
    ########################### Device info
    df_result['DeviceInfo'] = df['DeviceInfo'].fillna('unknown_device').str.lower()
    df_result['DeviceInfo_device'] = df_result['DeviceInfo'].apply(lambda x: ''.join([i for i in x if i.isalpha()]))
    df_result['DeviceInfo_version'] = df_result['DeviceInfo'].apply(lambda x: ''.join([i for i in x if i.isnumeric()]))

    ########################### Device info 2
    df_result['id_30'] = df['id_30'].fillna('unknown_device').str.lower()
    df_result['id_30_device'] = df_result['id_30'].apply(lambda x: ''.join([i for i in x if i.isalpha()]))
    df_result['id_30_version'] = df_result['id_30'].apply(lambda x: ''.join([i for i in x if i.isnumeric()]))

    ########################### Browser
    df_result['id_31'] = df['id_31'].fillna('unknown_device').str.lower()
    df_result['id_31_device'] = df_result['id_31'].apply(lambda x: ''.join([i for i in x if i.isalpha()]))

    df_result.to_feather(output_dir)


def make_emaildomain(df, output_dir):
    p = 'P_emaildomain'
    r = 'R_emaildomain'
    uknown = 'email_not_provided'

    df_result = pd.DataFrame()
    df_result[p] = df[p].fillna(uknown)
    df_result[r] = df[r].fillna(uknown)

    # Check if P_emaildomain matches R_emaildomain
    df_result['email_check'] = np.where((df_result[p] == df_result[r]) & (df_result[p] != uknown), 1, 0)

    df_result[p + '_prefix'] = df_result[p].apply(lambda x: x.split('.')[0])
    df_result[r + '_prefix'] = df_result[r].apply(lambda x: x.split('.')[0])

    df_result.to_feather(output_dir)

def make_C_D_feature(df, mode):
    for d_col in ["D1", "D12"]:
        agg_cols = []
        agg_cols.extend(["C{}".format(x) for x in range(1, 14+1)])
        agg_cols.extend(["D5"])
        for agg_col in agg_cols:
            to_feather(df[agg_col] / df[d_col], col_name="div_{}_{}".format(agg_col, d_col), mode=mode)

def make_id_countencode(df, mode):
    """
    idは、train/test別々でcountencodingする (バージョンとかOSとか流行り廃りを表現したい)
    :param df:
    :param output_dir:
    :return:
    """

    cols = ["id_{}".format(x) for x in np.arange(12, 38+1)]
    for col in cols:
        to_feather(df.groupby(col)["TransactionID"].transform("count") / len(df),
                   col_name="tra-tes_countencoding_{}".format(col), mode=mode)

df_train = pd.read_feather("../../data/original/train_all.feather")
df_test = pd.read_feather("../../data/original/test_all.feather")
make_identity_feature(df_train, "../../data/basic_feature/train/identity.feature")
make_identity_feature(df_test, "../../data/basic_feature/test/identity.feature")
make_emaildomain(df_train, "../../data/basic_feature/train/emaildomain.feature")
make_emaildomain(df_test, "../../data/basic_feature/test/emaildomain.feature")
make_C_D_feature(df_train, "train")
make_C_D_feature(df_test, "test")
make_id_countencode(df_train, "train")
make_id_countencode(df_test, "test")
