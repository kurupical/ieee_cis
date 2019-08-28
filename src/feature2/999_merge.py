import pandas as pd
import numpy as np
import gc
import glob
from tqdm import tqdm
import time

# time.sleep(60*60*2)
merge_features = []
merge_features = glob.glob("../../data/101_agg_id/train/*.feather")
merge_features.extend(glob.glob("../../data/102_countencoding/train/*.feather"))
merge_features = [x.replace("train", "{}") for x in merge_features]

df_train = pd.read_feather("../../data/baseline/train/baseline.feather")
# print(f)
dfs = [df_train]
dfs.extend([pd.read_feather(x.format("train")) for x in merge_features])
df_train = pd.concat(dfs, axis=1)

print("test")
df_test = pd.read_feather("../../data/baseline/test/baseline.feather")
dfs = [df_test]
dfs.extend([pd.read_feather(x.format("test")) for x in merge_features])
df_test = pd.concat(dfs, axis=1)

print(df_train)
# df_train, df_test = drop_minority_card123(df_train, df_test)
# df_train, df_test = valid_card(df_train, df_test)

drop_cols = [x for x in df_train.columns if x[:6] in ["TEMP__"]]
print("drop_cols: {}".format(drop_cols))
drop_cols.extend(["TransactionDT", "id_30", "id_31", "id_33"])
df_train = df_train.drop(drop_cols, axis=1)
df_test = df_test.drop(drop_cols, axis=1)
df_train.to_feather("../../data/merge/train_merge.feather")
df_test.to_feather("../../data/merge/test_merge.feather")
