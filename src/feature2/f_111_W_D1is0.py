
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import os

def apply_PCA(df_train, df_test, target_cols, n_components):
    w_ary = pd.concat([df_train[target_cols], df_test[target_cols]]).fillna(0).values

    w_ary = StandardScaler().fit_transform(w_ary)
    pca_ary = PCA(n_components=n_components).fit_transform(w_ary)

    col_names = ["W_D1is0_PCA_top100_{}".format(x) for x in range(n_components)]
    for i in range(len(col_names)):
        df_train[col_names[i]] = pca_ary[:len(df_train), i]
        df_test[col_names[i]] = pca_ary[len(df_train):, i]

    ret_cols = ["TransactionID"] + col_names
    return df_train[ret_cols], df_test[ret_cols]

def main():
    df_train = pd.read_feather("../../data/original/train_all.feather")
    df_test = pd.read_feather("../../data/original/test_all.feather")

    w_df_train = df_train.query("ProductCD == 'W' and D1 == 0")
    w_df_test = df_test.query("ProductCD == 'W' and D1 == 0")
    original_features = df_train.columns
    target_cols = ['TransactionAmt', 'V70', 'V69', 'V30', 'C5', 'V29', 'V91', 'V90',
                   'V306', 'V308', 'V307', 'D3', 'V282', 'V283', 'V279', 'V280',
                   'V128', 'V126', 'V95', 'V97', 'V127', 'V96', 'V49', 'V48', 'V313',
                   'V309', 'V315', 'V314', 'V310', 'V312', 'V281', 'D2', 'V288',
                   'V289', 'V284', 'V287', 'V285', 'D5', 'V11', 'V10', 'V131', 'V100',
                   'C14', 'C13', 'V129', 'V98', 'V317', 'V316', 'V318', 'V293',
                   'V294', 'V295', 'V133', 'V102', 'V130', 'V99', 'V134', 'V103',
                   'V132', 'V101', 'V45', 'V290', 'V292', 'V291',
                   'V36', 'V52', 'V38', 'dist1', 'V47', 'V51', 'V136', 'V7', 'V44',
                   'V83', 'V87', 'V105', 'V46', 'D15', 'V35', 'V76', 'V37', 'V137',
                   'V311', 'V135', 'V286', 'C12', 'D10', 'V106', 'V104', 'V5', 'D4',
                   'V43', 'V42']

    w_df_train, w_df_test = apply_PCA(w_df_train, w_df_test,
                                      target_cols=target_cols,
                                      n_components=8)


    df_train = pd.merge(df_train, w_df_train, how="left", on="TransactionID")
    df_test = pd.merge(df_test, w_df_test, how="left", on="TransactionID")
    df_train = df_train[[x for x in df_train.columns if x not in original_features]]
    df_test = df_test[[x for x in df_test.columns if x not in original_features]]

    print("train shape: {}".format(df_train.shape))
    print(df_train.head(5))
    print("test shape: {}".format(df_test.shape))
    print(df_test.tail(5))

    assert len(w_df_train) == df_train["W_D1is0_PCA_top100_0"].notnull().sum()
    df_train.to_feather("../../data/111_W_D1is0/train/specialfeat.feather")
    df_test.to_feather("../../data/111_W_D1is0/test/specialfeat.feather")

if __name__ == "__main__":
    print(os.path.basename(__file__))
    main()