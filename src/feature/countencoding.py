import pandas as pd

def countencoding(df_train, df_test, output_dir="../../data/countencoding"):

    df_all = pd.concat([df_train, df_test])
    remove_col = ["TransactionID", "TransactionDT", "isFraud"]

    for col in df_all.columns:
        print(col)
        if col in remove_col:
            continue
        df_result = pd.DataFrame()
        col_name = "countencoding_{}".format(col)
        df_result[col_name] = df_all[["TransactionID", col]].fillna("NAN").groupby(col).transform("count").values.reshape(-1)
        print(df_result.head(5))
        df_result.head(len(df_train)).reset_index(drop=True).to_feather("{}/train/{}.feather".format(output_dir, col_name))
        df_result.tail(len(df_test)).reset_index(drop=True).to_feather("{}/test/{}.feather".format(output_dir, col_name))


print("train")
df_train = pd.read_feather("../../data/original/train_all.feather")
print("test")
df_test = pd.read_feather("../../data/original/test_all.feather")

countencoding(df_train, df_test)
