import pandas as pd

def get_amount_decimal(df, output_path):
    def get_decimal(hoge):
        return len(str(hoge).split('.')[1])

    df_result = pd.DataFrame()
    df_result["TransactionAmt_decimal"] = df["TransactionAmt"].map(get_decimal)
    print(df_result.head(5))
    df_result.to_feather(output_path)

df_train = pd.read_csv("../../data/original/train_transaction.csv")
df_test = pd.read_csv("../../data/original/test_transaction.csv")

get_amount_decimal(df_train, output_path="../../data/decimal/train/amount_decimal.feather")
get_amount_decimal(df_test, output_path="../../data/decimal/test/amount_decimal.feather")