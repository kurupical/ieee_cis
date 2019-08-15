import pandas as pd
import lightgbm as lgb
import os
import numpy as np
from datetime import datetime as dt
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
import copy
from tqdm import tqdm
import gc
import time
from src.feature.common import reduce_mem_usage
import glob
from itertools import combinations

from src.feature.between_feature import *
from src.feature.aggregate import *

# hyper parameters
n_folds = 5
random_state = 0
n_loop = 1
target_col = "isFraud"
id_col = "TransactionID"
remove_cols = []
nrows_train = None # Noneだと全データをロード
nrows_test = None # Noneだと全データをロード
is_reduce_memory = True
# select_cols = None # 全てのcolumnを選ぶ
# select_cols = pd.read_csv("cols.csv")["column"].values

params = {'num_leaves': 60,
          'min_child_samples': 200,
          'objective': 'binary',
          'max_depth': 10,
          'learning_rate': 0.2,
          "boosting_type": "gbdt",
          "subsample": 0.5,
          "bagging_seed": 11,
          "metric": 'auc',
          "verbosity": -1,
          'reg_alpha': 1,
          'reg_lambda': 1,
          'colsample_bytree': 0.05,
          'early_stopping_rounds': 200,
          'n_estimators': 10000,
          }

def _get_categorical_features(df):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    feats = [col for col in list(df.columns) if df[col].dtype not in numerics]

    cat_cols = []
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

def learning(df_train, df_test, output_dir, output_top=300):

    cat_feats = _get_categorical_features(df_test)

    for f in cat_feats:
        # print(f)
        df_train[f] = df_train[f].fillna("nan").astype(str)
        df_test[f] = df_test[f].fillna("nan").astype(str)
        le = LabelEncoder().fit(df_train[f].append(df_test[f]))
        df_train[f] = le.transform(df_train[f])
        df_test[f] = le.transform(df_test[f])

    i = 0
    folds = StratifiedKFold(n_splits=n_folds, random_state=random_state)

    print("-----------------------------------")
    print("LOOP {} / {}".format(i+1, n_loop))
    print("-----------------------------------")

    df_pred_train = pd.DataFrame()
    df_pred_test = pd.DataFrame()
    print(df_test)
    df_pred_test[id_col] = df_test[id_col]
    df_importance = pd.DataFrame()
    df_result = pd.DataFrame()
    df_importance["column"] = df_train.drop([id_col, target_col, *remove_cols], axis=1).columns
    for n_fold, (train_idx, val_idx) in enumerate(folds.split(df_train, y=df_train[target_col])):
        if n_fold > 1:
            break
        print("-----------------------------------")
        print("Fold {} / {}".format(n_fold+1, n_folds))
        print("-----------------------------------")
        # X = df_train.drop([id_col, target_col, *remove_cols], axis=1)
        # y = df_train[target_col]

        lgb_train = lgb.Dataset(data=df_train.drop([id_col, target_col, *remove_cols], axis=1).iloc[train_idx], label=df_train[target_col].iloc[train_idx])
        lgb_val = lgb.Dataset(data=df_train.drop([id_col, target_col, *remove_cols], axis=1).iloc[val_idx], label=df_train[target_col].iloc[val_idx])

        model = lgb.train(copy.copy(params),
                          lgb_train,
                          categorical_feature=cat_feats,
                          valid_sets=[lgb_train, lgb_val],
                          verbose_eval=100)

        w_pred_train = model.predict(df_train.drop([id_col, target_col, *remove_cols], axis=1).iloc[val_idx])
        df_pred_train = df_pred_train.append(pd.DataFrame(
            {id_col: df_train[id_col].iloc[val_idx],
             "pred": w_pred_train,
             "y": df_train[target_col].iloc[val_idx]}
        ), ignore_index=True)
        w_pred_test = model.predict(df_test.drop([id_col, *remove_cols], axis=1))
        df_pred_test["pred_fold{}_{}".format(i, n_fold)] = w_pred_test
        df_importance["fold{}_{}_gain".format(i, n_fold)] = \
            model.feature_importance(importance_type="gain") / model.feature_importance(importance_type="gain").sum()
        df_importance["fold{}_{}_split".format(i, n_fold)] = \
            model.feature_importance(importance_type="split") / model.feature_importance(importance_type="split").sum()
        df_result = df_result.append(
            pd.DataFrame(
                {"fold": [n_fold],
                 "random_state": [random_state],
                 "auc_train": [roc_auc_score(df_train[target_col].iloc[train_idx], model.predict(df_train.drop([id_col, target_col, *remove_cols], axis=1).iloc[train_idx]))],
                 "auc_test": [roc_auc_score(df_train[target_col].iloc[val_idx], model.predict(df_train.drop([id_col, target_col, *remove_cols], axis=1).iloc[val_idx]))]}
            ),
            ignore_index=True
        )
        del w_pred_train, w_pred_test, lgb_train, lgb_val, model
        gc.collect()

    # output top features
    df_importance["total_importance"] = df_importance.sum(axis=1)
    for col in df_importance.sort_values("total_importance", ascending=False)["column"].iloc[:output_top]:
        df_train[[col]].to_feather("{}/train_{}.feather".format(output_dir, col))
        df_test[[col]].to_feather("{}/test_{}.feather".format(output_dir, col))
    df_submit = pd.DataFrame()
    df_submit[id_col] = df_test[id_col]
    df_submit[target_col] = df_pred_test.drop(id_col, axis=1).mean(axis=1)
    return df_submit, df_pred_train, df_pred_test, df_importance, df_result


# print("waiting...")
# time.sleep(60*60*3)
output_dir = "../../output/{}_extract".format(dt.now().strftime("%Y%m%d%H%M%S"))
os.makedirs("{}/feathers".format(output_dir))

df_submit = pd.read_csv("../../data/original/sample_submission.csv")
df_pred_train = pd.DataFrame()
df_pred_test = pd.DataFrame()
df_importance = pd.DataFrame()
df_importance_top = pd.DataFrame()

df_train_org = reduce_mem_usage(pd.read_feather("../../data/original/train_all.feather"))
df_test_org = reduce_mem_usage(pd.read_feather("../../data/original/test_all.feather"))

# 対象のfeatherを指定
target_features = ["C{}".format(x) for x in range(1, 14)]
target_features.extend(["V{}".format(x) for x in range(1, 339 + 1)])
target_features.extend(["D{}".format(x) for x in range(1, 15 + 1)])
target_features.extend(["id_{0:02d}".format(x) for x in range(1, 11 + 1)])
target_features.extend(["TransactionAmt"])

def load_feather(df_train, df_test, load_dir):
    return pd.read_feather(load_dir.format("train")), pd.read_feather(load_dir.format("test"))

funcs = []
# funcs.extend(make_list_2features(diff_2feature, target_features))
# funcs.extend(make_list_2features(div_2feature, target_features))
for f in glob.glob("../../output/div+agg20190815082430_extract/feathers/train*.feather"):
    funcs.append({"func": load_feather,
                  "col_name": os.path.basename(f.replace(".feather", "")),
                  "params": {
                      "load_dir": f.replace("train", "{}")
                  }})

# fa_divdiff = FeatureAggregateDivDiff(df_train_org, df_test_org)
# funcs.extend(fa_divdiff.feature_engineering_list)

np.random.shuffle(funcs)

extract_time = len(funcs) // 1000

print("all-features: {}".format(len(funcs)))
for i in range(extract_time):
    df_train = df_train_org[["TransactionID", "isFraud"]]
    df_test = df_test_org[["TransactionID"]]

    start_idx = len(funcs) * i // extract_time
    end_idx = len(funcs) * (i + 1) // extract_time
    target_funcs = funcs[start_idx:end_idx]

    print("load")
    for func in tqdm(target_funcs):
        df_train[func["col_name"]], df_test[func["col_name"]] = func["func"](df_train_org, df_test_org, **func["params"])

    sub, pred_train, pred_test, imp, result= learning(df_train=df_train,
                                                      df_test=df_test,
                                                      output_dir="{}/feathers".format(output_dir))

    imp["extract_time"] = extract_time
    df_submit += (sub / extract_time)
    # df_pred_train = df_pred_train.append(pred_train, ignore_index=True)
    # df_pred_test = df_pred_test.append(pred_test, ignore_index=True)
    df_importance = df_importance.append(imp)
    imp.to_csv("{}/importance_{}.csv".format(output_dir, i), index=False)
    sub.to_csv("{}/submit_{}.csv".format(output_dir, i), index=False)
    result.to_csv("{}/result_{}.csv".format(output_dir, i), index=False)

df_submit.to_csv("{}/submit.csv".format(output_dir), index=False)
# df_pred_train.to_csv("{}/predict_train.csv".format(output_dir), index=False)
# df_pred_test.to_csv("{}/submit_detail.csv".format(output_dir), index=False)
result.to_csv("{}/result.csv".format(output_dir), index=False)
df_importance.to_csv("{}/importance.csv".format(output_dir), index=False)