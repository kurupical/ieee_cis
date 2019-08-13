import pandas as pd
import numpy as np
import pickle
from sklearn.decomposition import PCA, TruncatedSVD
from src.feature.common import reduce_mem_usage

def weekday_basic(df, output_dir):
    print("weekday_basic start output={}".format(output_dir))
    cols = ["D{}".format(x) for x in range(1, 15 + 1)]
    for col in df[cols]:
        df_result = pd.DataFrame()
        fe_col = "{}_mod7".format(col)
        nan_idx = df[df[col].isnull()].index
        zero_idx = df[df[col] == 0].index
        df_result[fe_col] = (df[col].round(0) % 7).astype(str)
        df_result[fe_col].iloc[zero_idx] = "zero"
        df_result[fe_col].iloc[nan_idx] = "nan"
        df_result.to_feather("{}/{}.feather".format(output_dir, fe_col))

def aggregate_basic(df, output_path):
    print("aggregate_basic start output={}".format(output_path))
    df_result = pd.DataFrame()
    cols = ["D{}".format(x) for x in range(1, 15 + 1)]
    df_result["D_isnan_sum"] = df[cols].replace("nan", np.nan).isnull().sum(axis=1)
    df_result["D_iszero_sum"] = df[cols].replace("zero", np.nan).isnull().sum(axis=1)

    df_result["D5-6-7-8_nancount"] = df[["D6", "D7", "D8"]].isnull().sum(axis=1)
    df_result["D5-6-7-8_zerocount"] = (df[["D6", "D7", "D8"]] == 0).sum(axis=1)

    df_result["D12-13-14_nancount"] = df[["D12", "D13", "D14"]].isnull().sum(axis=1)
    df_result["D12-13-14_zerocount"] = (df[["D12", "D13", "D14"]] == 0).sum(axis=1)

    df_result = reduce_mem_usage(df_result, mode="output")
    df_result.to_feather(output_path)

    return df

def weekday_decomp(input_path_train, input_path_test, output_path_train, output_path_test):

    def datemod_ohe(df):
        # Dx_mod7をonehotencoding
        datemod_ary = []
        for col in ["D{}_mod7".format(x) for x in range(1, 15 + 1)]:
            ary = []
            for val in ["0", "1", "2", "3", "4", "5", "6", "zero", "nan"]:
                ary.append((df[col] == val).astype(int).values)
            datemod_ary.append(ary)

        datemod_ary = np.array(datemod_ary).reshape(-1, len(df)).transpose()
        return datemod_ary

    df_train = pd.read_feather(input_path_train)
    df_test = pd.read_feather(input_path_test)

    ohe_train = datemod_ohe(df_train)
    ohe_test = datemod_ohe(df_test)

    df_result_train = pd.DataFrame()
    df_result_test = pd.DataFrame()

    # PCA
    print("PCA")
    pca = PCA(n_components=5)
    pca.fit(ohe_train)
    ohe_train_decomp = pca.transform(ohe_train)
    ohe_test_decomp = pca.transform(ohe_test)
    for i in range(5):
        df_result_train["D_mod7_pca{}".format(i)] = ohe_train_decomp[:, i]
        df_result_test["D_mod7_pca{}".format(i)] = ohe_test_decomp[:, i]

    # SVD
    print("SVD")
    svd = TruncatedSVD(n_components=5)
    svd.fit(ohe_train)
    ohe_train_decomp = pca.transform(ohe_train)
    ohe_test_decomp = pca.transform(ohe_test)
    for i in range(5):
        df_result_train["D_mod7_svd{}".format(i)] = ohe_train_decomp[:, i]
        df_result_test["D_mod7_svd{}".format(i)] = ohe_test_decomp[:, i]

    print("df_result_train.shape : {}".format(df_result_train.shape))
    print("df_result_test.shape : {}".format(df_result_test.shape))
    df_result_train = reduce_mem_usage(df_result_train, mode="output")
    df_result_test = reduce_mem_usage(df_result_test, mode="output")

    df_result_train.to_feather(output_path_train)
    df_result_test.to_feather(output_path_test)

df_train = pd.read_feather("../../data/original/train_transaction.feather")
df_test = pd.read_feather("../../data/original/test_transaction.feather")

weekday_basic(df_train, output_dir="../../data/date")
weekday_basic(df_test, output_dir="../../data/date")
aggregate_basic(df_train, output_path="../../data/date/aggregate_basic_train.feather")
aggregate_basic(df_test, output_path="../../data/date/aggregate_basic_test.feather")

weekday_decomp(input_path_train="../../data/date/weekday_basic_train.feather",
               input_path_test="../../data/date/weekday_basic_test.feather",
               output_path_train="../../data/date/weekday_decomp_train.feather",
               output_path_test="../../data/date/weekday_decomp_test.feather")