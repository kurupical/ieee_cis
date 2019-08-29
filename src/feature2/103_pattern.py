import pandas as pd
import numpy as np
import sys
import tqdm

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

def make_nunique(df):
    def nunique(x):
        return len(np.unique(x.values))

    print(sys._getframe().f_code.co_name)
    df["nunique_D1-D15"] = df[["D{}".format(x) for x in range(1, 15+1)]].apply(nunique, axis=1)
    df["nunique_C1-C14"] = df[["C{}".format(x) for x in range(1, 14+1)]].apply(nunique, axis=1)
    v126_137 = ["V{}".format(x) for x in range(126, 137+1)]
    v202_216 = ["V{}".format(x) for x in range(202, 216+1)]
    v263_278 = ["V{}".format(x) for x in range(263, 278+1)]
    v306_321 = ["V{}".format(x) for x in range(306, 321+1)]

    for agg_type in tqdm.tqdm(["mean", "std"]):
        df["{}_V126-V137".format(agg_type)] = df[v126_137].apply(agg_type, axis=1)
        df["{}_V202-V216".format(agg_type)] = df[v202_216].apply(agg_type, axis=1)
        df["{}_V263-V278".format(agg_type)] = df[v263_278].apply(agg_type, axis=1)
        df["{}_V306-V321".format(agg_type)] = df[v306_321].apply(agg_type, axis=1)
        df["{}_V126-V321_money".format(agg_type)] = df[v126_137+v202_216+v263_278+v306_321].apply(agg_type, axis=1)

        if agg_type in ["mean", "std"]:
            for cs in [v126_137, v202_216, v263_278, v306_321]:
                for c in cs:
                    df["div_{}_aggVs".format(c)] = df[c] / df[cs].apply(agg_type, axis=1)
        for i in range(15):
            v1 = 202+i
            if i <= 5:
                v2 = 263+i
                v3 = 306+i
            else:
                v2 = 264+i
                v3 = 307+i
            df["{}_V{}_{}_{}".format(agg_type, v1, v2, v3)] = df[["V{}".format(v1), "V{}".format(v2), "V{}".format(v3)]].apply(agg_type, axis=1)
        for i in range(12):
            v1 = 126+i
            if i <= 5:
                v2 = 202+i
                df["{}_V{}_{}".format(agg_type, v1, v2)] = df[["V{}".format(v1), "V{}".format(v2)]].apply(agg_type, axis=1)
            else:
                v2 = 205+i
                df["{}_V{}_{}".format(agg_type, v1, v2)] = df[["V{}".format(v1), "V{}".format(v2)]].apply(agg_type, axis=1)
                if i <= 8:
                    v2 = 202+i
                    df["{}_V{}_{}".format(agg_type, v1, v2)] = df[["V{}".format(v1), "V{}".format(v2)]].apply(agg_type, axis=1)

    return df

df_train = pd.read_feather("../../data/baseline/train/baseline.feather")
df_test = pd.read_feather("../../data/baseline/test/baseline.feather")

original_features = df_train.columns


df_train = make_nan_pattern(df_train)
df_test = make_nan_pattern(df_test)

df_train = make_nunique(df_train)
df_test = make_nunique(df_test)

df_train = df_train[[x for x in df_train.columns if x not in original_features]]
df_test = df_test[[x for x in df_test.columns if x not in original_features]]

print("train shape: {}".format(df_train.shape))
print(df_train.head(5))
print(df_train.describe())
print("test shape: {}".format(df_test.shape))
print(df_test.head(5))
print(df_test.describe())


df_train.to_feather("../../data/103_pattern/train/pattern.feather")
df_test.to_feather("../../data/103_pattern/test/pattern.feather")
