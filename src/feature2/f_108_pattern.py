import pandas as pd
import numpy as np
import tqdm
import os
import datetime as dt

def max_values_and_col(df):
    def _get_col(df, cols, col_name):
        """
        グループのなかで最大/最小の項目名（最大の項目が複数ある場合は全部）取得
        :param df:
        :param cols:
        :param col_name:
        :return:
        """
        max_col_names = []
        for is_target_cols in (df[cols].values - df[col_name].values.reshape(-1, 1) == 0):
            w_max_col_name = ""
            for i, is_target in enumerate(is_target_cols):
                if is_target:
                    w_max_col_name += cols[i]
            max_col_names.append(w_max_col_name)
        return max_col_names

    def _get_value_and_col(df, cols, new_col_name, div_amt=False):
        print(new_col_name)
        df["{}_max".format(new_col_name)] = df[cols].max(axis=1)
        df["{}_maxcol".format(new_col_name)] = _get_col(df, cols, col_name="{}_max".format(new_col_name))
        if div_amt:
            df["div_{}_max_AMT".format(new_col_name)] = df["TransactionAmt"] / df["{}_max".format(new_col_name)]

        df["{}_min".format(new_col_name)] = df[cols].min(axis=1)
        df["{}_mincol".format(new_col_name)] = _get_col(df, cols, col_name="{}_max".format(new_col_name))
        if div_amt:
            df["div_{}_min_AMT".format(new_col_name)] = df["TransactionAmt"] / df["{}_min".format(new_col_name)]

        df["diff_{}_max_min".format(new_col_name)] = df["{}_max".format(new_col_name)] - df["{}_min".format(new_col_name)]
        # 項目プレビュー
        if "isFraud" in df.columns:
            print(df[["{}_maxcol".format(new_col_name), "isFraud"]].groupby("{}_maxcol".format(new_col_name)).agg({"isFraud": ["count", "mean"]}))
        return df

    df = _get_value_and_col(df, cols=["D1", "D2", "D10", "D11", "D12", "D13", "D14", "D15"],
                            new_col_name="D1_2_10_11_12_13_14_15")
    df = _get_value_and_col(df, cols=["D3", "D4", "D5", "D6", "D7"],
                            new_col_name="D3_4_5_6_7")
    df = _get_value_and_col(df, cols=["C{}".format(x) for x in range(1, 14+1)],
                            new_col_name="allC")

    # money
    df = _get_value_and_col(df, cols=["V{}".format(x) for x in range(126, 137+1)],
                            new_col_name="V126_137", div_amt=True)
    df = _get_value_and_col(df, cols=["V{}".format(x) for x in range(202, 216+1)],
                            new_col_name="V202_216", div_amt=True)
    df = _get_value_and_col(df, cols=["V{}".format(x) for x in range(263, 278+1)],
                            new_col_name="V263_V278", div_amt=True)
    df = _get_value_and_col(df, cols=["V{}".format(x) for x in range(306, 321+1)],
                            new_col_name="V306_321", div_amt=True)

    # other
    df = _get_value_and_col(df, cols=["V144", "V181", "V297", "V328"],
                            new_col_name="V144_181_297_328")
    df = _get_value_and_col(df, cols=["V95", "V101", "V143", "V167", "V177", "V279", "V293"],
                            new_col_name="V95_101_143_167_177_279_293")
    df = _get_value_and_col(df, cols=["V145", "V183", "V299"],
                            new_col_name="V145_183_299")
    df = _get_value_and_col(df, cols=["V102", "V178", "V294"],
                            new_col_name="V102_V178_V294")
    df = _get_value_and_col(df, cols=["V96", "V323"],
                            new_col_name="V96_V323")
    df = _get_value_and_col(df, cols=["V97", "V103", "V280", "V295", "V324"],
                            new_col_name="V97_103_280_295_324")
    df = _get_value_and_col(df, cols=["V{}".format(x) for x in range(167, 201+1)],
                            new_col_name="V167-201")
    df = _get_value_and_col(df, cols=["id_01", "id_06", "id_08", "id_10"],
                            new_col_name="id_01_06_08_10")
    df = _get_value_and_col(df, cols=["id_03", "id_05", "id_07", "id_09"],
                            new_col_name="id_03_05_07_09")

    for i in range(15):
        v1 = 202 + i
        if i <= 5:
            v2 = 263 + i
            v3 = 306 + i
        else:
            v2 = 264 + i
            v3 = 307 + i
        col_name = "V{}_{}_{}".format(v1, v2, v3)
        df = _get_value_and_col(df,
                                cols=["V{}".format(v1), "V{}".format(v2), "V{}".format(v3)],
                                new_col_name=col_name)
    for i in range(12):
        v1 = 126 + i
        if i <= 5:
            v2 = 202 + i
            v3 = 263 + i
            v4 = 306 + i
            col_name = "V{}_{}_{}_{}".format(v1, v2, v3, v4)
            df = _get_value_and_col(df,
                                    cols=["V{}".format(v1), "V{}".format(v2), "V{}".format(v3), "V{}".format(v4)],
                                    new_col_name=col_name)
        else:
            v2 = 205 + i
            v3 = 267 + i
            v4 = 310 + i
            col_name = "V{}_{}_{}_{}".format(v1, v2, v3, v4)
            df = _get_value_and_col(df,
                                    cols=["V{}".format(v1), "V{}".format(v2), "V{}".format(v3), "V{}".format(v4)],
                                    new_col_name=col_name)
            if i <= 8:
                v2 = 202 + i
                v3 = 264 + i
                v4 = 307 + i
                col_name = "V{}_{}_{}_{}".format(v1, v2, v3, v4)
                df = _get_value_and_col(df,
                                        cols=["V{}".format(v1), "V{}".format(v2), "V{}".format(v3), "V{}".format(v4)],
                                        new_col_name=col_name)

    return df

def ProductCD_M_pattern_Wonly(df):
    df["ProductW_Mpattern"] = df["M1"].astype(str) + df["M2"].astype(str) + df["M3"].astype(str) + \
                              df["M4"].astype(str) + df["M5"].astype(str) + df["M6"].astype(str) + \
                              df["M7"].astype(str) + df["M8"].astype(str) + df["M9"].astype(str)
    df["ProductW_Mpattern"][df["ProductCD"] != "W"] = np.nan
    return df

def pattern_AMT_Decimal_keta(df):
    cols = ["ProductCD",
            "M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8", "M9",
            # "card1", "card2", "card3", "card4", "card5", "card6",
            "addr1", "addr2"
            ]
    for col in cols:
        df["{}_AMT_Decimal_keta".format(col)] = df[col].astype(str) + df["AMT_Decimal_keta"].astype(str)
    return df

def main():

    # 元データのロード
    df_train = pd.read_feather("../../data/baseline/train/baseline.feather")
    df_test = pd.read_feather("../../data/baseline/test/baseline.feather")

    original_features = df_train.columns

    df_train = pattern_AMT_Decimal_keta(df_train)
    df_test = pattern_AMT_Decimal_keta(df_test)

    df_train = ProductCD_M_pattern_Wonly(df_train)
    df_test = ProductCD_M_pattern_Wonly(df_test)

    df_train = max_values_and_col(df_train)
    df_test = max_values_and_col(df_test)

    df_train = df_train[[x for x in df_train.columns if x not in original_features]]
    df_test = df_test[[x for x in df_test.columns if x not in original_features]]

    df_train.reset_index(drop=True).to_feather("../../data/108_pattern/train/pattern.feather")
    df_test.reset_index(drop=True).to_feather("../../data/108_pattern/test/pattern.feather")

if __name__ == "__main__":
    print(os.path.basename(__file__))
    main()
