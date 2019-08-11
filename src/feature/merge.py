import pandas as pd
import gc
import glob
from tqdm import tqdm

merge_features = [
    "../../data/date/aggregate_basic_{}.feather",
    "../../data/date/weekday_basic_{}.feather",
    "../../data/date/weekday_decomp_{}.feather"
]

merge_features.extend(glob.glob("../../data/agg/card1/train/*"))
# merge_features.extend(glob.glob("../../data/agg/card2/train/*"))

merge_features = [x.replace("train", "{}") for x in merge_features]
print("train")
df_train = pd.read_feather("../../data/original/train_all.feather")
for f in tqdm(merge_features):
    # print(f)
    df_train = pd.merge(df_train, pd.read_feather(f.format("train")), how="left")
df_train.to_feather("../../data/merge/train_merge.feather")

del df_train
gc.collect()

print("test")
df_test = pd.read_feather("../../data/original/test_all.feather")
for f in tqdm(merge_features):
    # print(f)
    df_test = pd.merge(df_test, pd.read_feather(f.format("test")), how="left")
df_test.to_feather("../../data/merge/test_merge.feather")