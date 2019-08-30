
import pandas as pd
import numpy as np
import tqdm

def countencoding(df_train, df_test, target_cols, agg_cols):
    for col in tqdm.tqdm(target_cols):
        temp_df = pd.concat([df_train[[col]], df_test[[col]]])
        fq_encode = temp_df[col].value_counts(dropna=False).to_dict()
        df_train["fq_enc_" + col] = df_train[col].map(fq_encode)
        df_test['fq_enc_' + col] = df_test[col].map(fq_encode)

    for col in agg_cols:
        temp_df = pd.concat([df_train[[col]], df_test[[col]]])
        fq_encode = temp_df[col].value_counts().to_dict()

        df_train["fq_enc_" + col + '_total'] = df_train[col].map(fq_encode)
        df_test["fq_enc_" + col + '_total'] = df_test[col].map(fq_encode)

    for agg_col in tqdm.tqdm(agg_cols):
        for col in target_cols:
            new_column = "fq_enc_(" + col + '+' + agg_col + ")"

            temp_df = pd.concat([df_train[[col, agg_col]], df_test[[col, agg_col]]])
            temp_df[new_column] = temp_df[col].astype(str) + '_' + (temp_df[agg_col]).astype(str)
            fq_encode = temp_df[new_column].value_counts().to_dict()

            df_train[new_column] = (df_train[col].astype(str) + '_' + df_train[agg_col].astype(str)).map(fq_encode)
            df_test[new_column] = (df_test[col].astype(str) + '_' + df_test[agg_col].astype(str)).map(fq_encode)

            df_train[new_column] /= df_train["fq_enc_" + agg_col + '_total']
            df_test[new_column] /= df_test["fq_enc_" + agg_col + '_total']

    return df_train, df_test

df_train = pd.read_feather("../../data/baseline/train/baseline.feather")
df_test = pd.read_feather("../../data/baseline/test/baseline.feather")

original_features = df_train.columns

target_cols = ['card1','card2','card3','card5',
               'C1','C2','C3','C4','C5','C6','C7','C8','C9','C10','C11','C12','C13','C14',
               'D1','D2','D3','D4','D5','D6','D7','D8',
               'addr1','addr2',
               'dist1','dist2',
               'P_emaildomain', 'R_emaildomain',
               'DeviceInfo','DeviceInfo_device','DeviceInfo_version',
               'id_30','id_30_device','id_30_version',
               'id_31_device',
               'id_33',
               'TEMP__uid','TEMP__uid2','TEMP__uid3', "TEMP__uid2+DT", "TEMP__uid3+DT", "TEMP__uid4",
               "TEMP__uid2+DT+M4", "TEMP__uid3+DT+M4"
              ]
df_train, df_test = countencoding(df_train, df_test,
                                  target_cols=target_cols,
                                  agg_cols=["TEMP__DT_M", "TEMP__DT_W", "TEMP__DT_D", "ProductCD"])

df_train = df_train[[x for x in df_train.columns if x not in original_features]]
df_test = df_test[[x for x in df_test.columns if x not in original_features]]

print("train shape: {}".format(df_train.shape))
print(df_train.head(5))
print(df_train.describe())
print("test shape: {}".format(df_test.shape))
print(df_test.head(5))
print(df_test.describe())


df_train.to_feather("../../data/102_countencoding/train/ce.feather")
df_test.to_feather("../../data/102_countencoding/test/ce.feather")
