import pandas as pd
from itertools import combinations
import numpy as np
from tqdm import tqdm

df = pd.read_feather("../../data/merge/train_merge.feather")
idx = np.arange(len(df))
train_idx = np.sort(np.random.choice(idx, 30000, replace=False))
df = df.iloc[train_idx]
df_imp = pd.read_csv("../../output/20190914225333/importance.csv", index_col=0)
df_imp["importance"] = df_imp.sum(axis=1)
df_imp = df_imp.sort_values("importance", ascending=False)
df = df[[x for x in df_imp["column"].values if x in df.columns]]

print(df.columns)
def _get_categorical_features(df):
    numerics = ['int8', 'int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    feats = [col for col in list(df.columns) if df[col].dtype not in numerics]

    cat_cols = []
    cat_cols.extend(["isFraud"])
    cat_cols.extend(["TransactionID"])
    cat_cols.extend(["ProductCD"])
    cat_cols.extend(["card{}".format(x) for x in np.arange(1, 6+1)])
    cat_cols.extend(["addr1", "addr2"])
    cat_cols.extend(["P_emaildomain", "R_emaildomain"])
    cat_cols.extend(["M{}".format(x) for x in np.arange(1, 9+1)])
    cat_cols.extend(["DeviceType", "DeviceInfo"])
    cat_cols.extend(["id_{}".format(x) for x in np.arange(12, 38+1)])

    cat_cols = [x for x in df.columns if x in cat_cols]
    feats.extend([x for x in cat_cols if x not in feats])
    return feats

df = df.corr().to_csv("corr.csv")
"""
cols = [x for x in df.columns if x not in _get_categorical_features(df)]
result = []
remove_cols = []

for cols in tqdm(combinations(df.columns, r=2)):
    if cols[0] in remove_cols or cols[1] in remove_cols:
        print("{} is already delete".format(cols[1]))
        continue
    else:
        result.append([cols[0], cols[1], df[list(cols)].corr().values[0, 1]])
        if df[list(cols)].corr().values[0, 1] < -0.95 or df[list(cols)].corr().values[0, 1] > 0.95:
            remove_cols.append(cols[1])

result = np.array(result)
df_result = pd.DataFrame()
df_result["col1"] = result[:, 0]
df_result["col2"] = result[:, 1]
df_result["corr"] = result[:, 2].astype(float)
df_result = df_result.query("corr < -0.9 or corr > 0.9")
"""
df_result.to_csv("corr.csv", index=False)
