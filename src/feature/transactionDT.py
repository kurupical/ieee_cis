import pandas as pd
import datetime as dt
pd.set_option('display.max_columns', 100)

def convDT(df_train, df_test, output_dir):
    def get_normalize_day(x):
        if x.month in [4, 6, 9]:
            return (x.day - 1) / 29
        if x.month in [2]:
            return (x.day - 1) / 27
        else:
            return (x.day - 1) / 30

    df = pd.concat([df_train, df_test]).reset_index(drop=True)
    df["TransactionDay"] = (df["TransactionDT"] / (60*60*24)).astype(int)
    df["TransactionDate"] = [dt.date(2017, 12, 1) + dt.timedelta(days=int(x)-1) for x in df["TransactionDay"].values]

    df_result = pd.DataFrame()
    weekday_ary = ["Thursday", "Friday", "Saturday", "Sunday", "Monday", "Tuesday", "Wednesday"]
    df_result["TransactionWeekday"] = [weekday_ary[x%7] for x in df["TransactionDay"].values]
    df_result["TransactionDay_normalize"] = [get_normalize_day(x) for x in df["TransactionDate"].values]

    df_result["TransactionDT_isDecember"] = [int(x.month==12) for x in df["TransactionDate"].values]
    df_result["TransactionDT_month_in_quarter"] = [x.month%3 for x in df["TransactionDate"].values]
    df_result["TransactionDT_month_in_half-yearly"] = [x.month%6 for x in df["TransactionDate"].values]
    df_result["TransactionDT_hhmmss"] = df["TransactionDT"] % (60*60*24)
    # df_result["TransactionDT_timerange"] = ["range_{}".format(x%6) for x in df_result["TransactionDT_hhmmss"]]


    for i in range(1, 15+1):
        df_result["diff_TransactionDay_D{}".format(i)] = df["D{}".format(i)] - df["TransactionDay"]
        # df_result["div_TransactionDay_D{}".format(i)] = df["D{}".format(i)] / df["TransactionDay"]

    df_result["ProductCD"] = df["ProductCD"]
    df_result["TransactionDay_ProductCD_countencoding"] = df.groupby(["TransactionDay", "ProductCD"])["TransactionID"].transform("count")
    df_result["TransactionWeek_ProductCD_countencoding"] = df_result.groupby(["TransactionWeekday", "ProductCD"])["ProductCD"].transform("count")
    df_result["TransactionQuarter_ProductCD_countencoding"] = df_result.groupby(["TransactionDT_month_in_quarter", "ProductCD"])["ProductCD"].transform("count")
    df_result["TransactionHalfyearly_ProductCD_countencoding"] = df_result.groupby(["TransactionDT_month_in_half-yearly", "ProductCD"])["ProductCD"].transform("count")
    df_result["TransactionDecember_ProductCD_countencoding"] = df_result.groupby(["TransactionDT_isDecember", "ProductCD"])["ProductCD"].transform("count")

    df_result = df_result.drop("ProductCD", axis=1)
    print("HEAD")
    print(df_result.head(5))
    print("TAIL")
    print(df_result.tail(5))
    df_result.head(len(df_train)).reset_index(drop=True).to_feather(output_dir.format("train"))
    df_result.tail(len(df_test)).reset_index(drop=True).to_feather(output_dir.format("test"))

df_train = pd.read_feather("../../data/original/train_all.feather")
df_test = pd.read_feather("../../data/original/test_all.feather")
convDT(df_train, df_test, output_dir="../../data/transactionDT/{}/convDT.feather")
