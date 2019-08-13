import pandas as pd
import gc
import glob
from tqdm import tqdm

merge_features = glob.glob("../../data/date/*train*.feather")

# merge_features.extend(glob.glob("../../data/agg/card2/train/*"))
important_feature = pd.read_csv("../../output/feature_extract(agg,div)_20190812195625/importance.csv")
merge_features.extend(["../../data/agg/train/{}.feather".format(x) for x in important_feature["column"][:1000]])

merge_features = [x.replace("train", "{}") for x in merge_features]
print("train")
df_train = pd.read_feather("../../data/original/train_all.feather")
# print(f)
dfs = [df_train]
dfs.extend([pd.read_feather(x.format("train")) for x in merge_features])
df_train = pd.concat(dfs, axis=1)
df_train.to_feather("../../data/merge/train_merge.feather")

del df_train
gc.collect()

print("test")
df_test = pd.read_feather("../../data/original/test_all.feather")
dfs = [df_test]
dfs.extend([pd.read_feather(x.format("test")) for x in merge_features])
df_test = pd.concat(dfs, axis=1)
df_test.to_feather("../../data/merge/test_merge.feather")