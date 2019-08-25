import pandas as pd

def to_feather(df, col_name, mode):
    pd.DataFrame(df.values, columns=[col_name]).reset_index(drop=True).to_feather(
        "../../data/countencoding/{}/{}.feather".format(mode, col_name))

def countencoding(df_train, df_test, output_dir="../../data/countencoding"):

    remove_col = ["TransactionID", "TransactionDT", "isFraud"]
    target_cols = ['card1', 'card2', 'card3', 'card5',
                  'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12', 'C13', 'C14',
                  'D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8',
                  'addr1', 'addr2',
                  'dist1', 'dist2',
                  'P_emaildomain', 'R_emaildomain',
                  'DeviceInfo']
    for col in target_cols:
        print(col)
        if col in remove_col:
            continue
        df_result = pd.DataFrame()
        col_name = "countencoding_{}".format(col)
        df_train[col] = df_train[col].fillna("NAN")
        enc = df_train.groupby(col)["TransactionID"].count().to_dict()

        print(df_train[col].map(enc).head(5))
        to_feather(df_train[col].map(enc), col_name=col, mode="train")
        to_feather(df_test[col].map(enc), col_name=col, mode="test")


print("train")
df_train = pd.read_feather("../../data/original/train_all.feather")
print("test")
df_test = pd.read_feather("../../data/original/test_all.feather")

countencoding(df_train, df_test)
