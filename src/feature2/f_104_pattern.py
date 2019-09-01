import pandas as pd
import numpy as np
import sys
import tqdm
import os

def join_string(x):
    return "".join(x.astype(str).values)

def make_similarV_feature(df):
    print(sys._getframe().f_code.co_name)
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
                df["div_transactionAMT_aggV{}-{}_{}".format(cs[0], cs[-1], agg_type)] = \
                    df[cs].apply(agg_type, axis=1) / df["TransactionAmt"]
                for c in cs:
                    df["div_{}_aggV{}-{}_{}".format(c, cs[0], cs[-1], agg_type)] = df[c] / df[cs].apply(agg_type, axis=1)
        for i in range(15):
            v1 = 202+i
            if i <= 5:
                v2 = 263+i
                v3 = 306+i
            else:
                v2 = 264+i
                v3 = 307+i
            col_name = "{}_V{}_{}_{}".format(agg_type, v1, v2, v3)
            df[col_name] = df[["V{}".format(v1), "V{}".format(v2), "V{}".format(v3)]].apply(agg_type, axis=1)
            df["div_V{}_{}".format(v1, col_name)] = df["V{}".format(v1)] / df[col_name]
            df["div_V{}_{}".format(v2, col_name)] = df["V{}".format(v2)] / df[col_name]
            df["div_V{}_{}".format(v3, col_name)] = df["V{}".format(v3)] / df[col_name]
        for i in range(12):
            v1 = 126+i
            if i <= 5:
                v2 = 202+i
                v3 = 263+i
                v4 = 306+i
                col_name = "{}_V{}_{}_{}_{}".format(agg_type, v1, v2, v3, v4)
                df[col_name] = df[["V{}".format(v1), "V{}".format(v2), "V{}".format(v3), "V{}".format(v4)]].apply(agg_type, axis=1)
                df["div_V{}_{}".format(v1, col_name)] = df["V{}".format(v1)] / df[col_name]
                df["div_V{}_{}".format(v2, col_name)] = df["V{}".format(v2)] / df[col_name]
                df["div_V{}_{}".format(v3, col_name)] = df["V{}".format(v3)] / df[col_name]
                df["div_V{}_{}".format(v4, col_name)] = df["V{}".format(v4)] / df[col_name]
            else:
                v2 = 205+i
                v3 = 267+i
                v4 = 310+i
                col_name = "{}_V{}_{}_{}_{}".format(agg_type, v1, v2, v3, v4)
                df[col_name] = df[["V{}".format(v1), "V{}".format(v2), "V{}".format(v3), "V{}".format(v4)]].apply(agg_type, axis=1)
                df["div_V{}_{}".format(v1, col_name)] = df["V{}".format(v1)] / df[col_name]
                df["div_V{}_{}".format(v2, col_name)] = df["V{}".format(v2)] / df[col_name]
                df["div_V{}_{}".format(v3, col_name)] = df["V{}".format(v3)] / df[col_name]
                df["div_V{}_{}".format(v4, col_name)] = df["V{}".format(v4)] / df[col_name]
                if i <= 8:
                    v2 = 202+i
                    v3 = 264+i
                    v4 = 307+i
                    col_name = "{}_V{}_{}_{}_{}".format(agg_type, v1, v2, v3, v4)
                    df[col_name] = df[["V{}".format(v1), "V{}".format(v2), "V{}".format(v3), "V{}".format(v4)]].apply(agg_type, axis=1)
                    df["div_V{}_{}".format(v1, col_name)] = df["V{}".format(v1)] / df[col_name]
                    df["div_V{}_{}".format(v2, col_name)] = df["V{}".format(v2)] / df[col_name]
                    df["div_V{}_{}".format(v3, col_name)] = df["V{}".format(v3)] / df[col_name]
                    df["div_V{}_{}".format(v4, col_name)] = df["V{}".format(v4)] / df[col_name]

    return df

def make_similarV_feature2(df):
    print(sys._getframe().f_code.co_name)

    def meanstd_diffdiv(group):
        df["mean_{}".format("_".join(group))] = df[group].mean(axis=1)
        df["std_{}".format("_".join(group))] = df[group].std(axis=1)
        for col in group:
            df["diff_mean_{}".format("_".join(group))] = df[col] - df["mean_{}".format("_".join(group))]
            df["div_mean_{}".format("_".join(group))] = df[col] / df["mean_{}".format("_".join(group))]
            df["div_std_{}".format("_".join(group))] = df[col] / df["std_{}".format("_".join(group))]

        return df

    groups = []
    groups.append(["V144", "V181", "V297", "V328"])
    groups.append(["V95", "V101", "V143", "V167", "V177", "V279", "V293"])
    groups.append(["V145", "V183", "V299"])
    groups.append(["V102", "V178", "V294"])
    groups.append(["V96", "V323"])
    groups.append(["V97", "V103", "V280", "V295", "V324"])

    for group in groups:
        df = meanstd_diffdiv(group)

    return df

def isDecember_feature(df):
    print(sys._getframe().f_code.co_name)
    df["DT_isDecember__ProductCD"] = df["DT_isDecember"].astype(str) + "_" + df["ProductCD"].astype(str)
    df["DT_isDecember__allM"] = df[["DT_isDecember"] + ["M{}".format(x) for x in range(1, 9+1)]].apply(join_string).astype(str)
    df["DT_isDecember__id01isnan"] = df["DT_isDecember"].astype(str) + "_" + df["id_01"].isnull().astype(int).astype(str)

    return df

def main():
    df_train = pd.read_feather("../../data/baseline/train/baseline.feather")
    df_test = pd.read_feather("../../data/baseline/test/baseline.feather")

    original_features = df_train.columns

    df_train = isDecember_feature(df_train)
    df_test = isDecember_feature(df_test)

    df_train = make_similarV_feature(df_train)
    df_test = make_similarV_feature(df_test)

    df_train = make_similarV_feature2(df_train)
    df_test = make_similarV_feature2(df_test)

    df_train = df_train[[x for x in df_train.columns if x not in original_features]]
    df_test = df_test[[x for x in df_test.columns if x not in original_features]]

    print("train shape: {}".format(df_train.shape))
    print(df_train.head(5))
    print(df_train.describe())
    print("test shape: {}".format(df_test.shape))
    print(df_test.head(5))
    print(df_test.describe())


    df_train.to_feather("../../data/104_pattern/train/pattern.feather")
    df_test.to_feather("../../data/104_pattern/test/pattern.feather")

if __name__ == "__main__":
    print(os.path.basename(__file__))
    main()
