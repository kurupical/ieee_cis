import pandas as pd
import lightgbm as lgb
import os
import numpy as np
from datetime import datetime as dt
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
import copy
from tqdm import tqdm
import gc
import time
from src.feature.common import reduce_mem_usage
import random

# hyper parameters
n_folds = 5
random_state = 0
n_loop = 1
target_col = "isTrain"
id_col = "TransactionID"
remove_cols = ["TransactionDT", "isFraud",
               "id_13", "id_30", "id_31"# train/testで分布が違いすぎる (別手法で対応)
               ]
nrows_train = None # Noneだと全データをロード
nrows_test = None # Noneだと全データをロード
is_reduce_memory = False
select_cols = None # 全てのcolumnを選ぶ
# select_cols = pd.read_csv("cols.csv")["column"].values

params = {'num_leaves': 60,
          'min_child_samples': 200,
          'objective': 'binary',
          'max_depth': 10,
          'learning_rate': 0.01,
          "boosting_type": "gbdt",
          "subsample": 0.6,
          "bagging_seed": 11,
          "metric": 'auc',
          "verbosity": -1,
          'reg_alpha': 1,
          'reg_lambda': 1,
          'colsample_bytree': 0.3,
          'early_stopping_rounds': 50,
          'n_estimators': 500,
          }

def _get_categorical_features(df):
    numerics = ['int8', 'int16', 'int32', 'int64', 'float16', 'float32', 'float64']
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

def learning(df_train):

    cat_feats = _get_categorical_features(df_train)

    for f in cat_feats:
        # print(f)
        df_train[f] = df_train[f].fillna("nan").astype(str)
        le = LabelEncoder().fit(df_train[f])
        df_train[f] = le.transform(df_train[f])

    i = 0
    folds = KFold(n_splits=n_folds, shuffle=True, random_state=0)

    print("-----------------------------------")
    print("LOOP {} / {}".format(i+1, n_loop))
    print("-----------------------------------")

    df_pred_train = pd.DataFrame()
    df_importance = pd.DataFrame()
    df_result = pd.DataFrame()
    df_importance["column"] = df_train.drop([id_col, target_col], axis=1).columns
    for n_fold, (train_idx, val_idx) in enumerate(folds.split(df_train)):
        print("-----------------------------------")
        print("Fold {} / {}".format(n_fold+1, n_folds))
        print("-----------------------------------")
        # X = df_train.drop([id_col, target_col, *remove_cols], axis=1)
        # y = df_train[target_col]
        if n_fold == 1: continue
        if n_fold == 2: continue
        if n_fold == 3: continue
        if n_fold == 4: continue

        lgb_train = lgb.Dataset(data=df_train.drop([id_col, target_col], axis=1).iloc[train_idx], label=df_train[target_col].iloc[train_idx])
        lgb_val = lgb.Dataset(data=df_train.drop([id_col, target_col], axis=1).iloc[val_idx], label=df_train[target_col].iloc[val_idx])

        model = lgb.train(copy.copy(params),
                          lgb_train,
                          categorical_feature=cat_feats,
                          valid_sets=[lgb_train, lgb_val],
                          verbose_eval=100)

        w_pred_train = model.predict(df_train.drop([id_col, target_col], axis=1).iloc[val_idx])
        df_pred_train = df_pred_train.append(pd.DataFrame(
            {id_col: df_train[id_col].iloc[val_idx],
             "pred": w_pred_train,
             "y": df_train[target_col].iloc[val_idx]}
        ), ignore_index=True)
        df_importance["fold{}_{}_gain".format(i, n_fold)] = \
            model.feature_importance(importance_type="gain") / model.feature_importance(importance_type="gain").sum()
        df_importance["fold{}_{}_split".format(i, n_fold)] = \
            model.feature_importance(importance_type="split") / model.feature_importance(importance_type="split").sum()
        df_result = df_result.append(
            pd.DataFrame(
                {"fold": [n_fold],
                 "random_state": [random_state],
                 "auc_train": [roc_auc_score(df_train[target_col].iloc[train_idx], model.predict(df_train.drop([id_col, target_col], axis=1).iloc[train_idx]))],
                 "auc_test": [roc_auc_score(df_train[target_col].iloc[val_idx], model.predict(df_train.drop([id_col, target_col], axis=1).iloc[val_idx]))]}
            ),
            ignore_index=True
        )
        del w_pred_train, lgb_train, lgb_val, model
        gc.collect()

    return df_pred_train, df_importance, df_result


# print("waiting...")
# time.sleep(60*60)
output_dir = "../../output/{}".format(dt.now().strftime("%Y%m%d%H%M%S"))
os.makedirs(output_dir)

df_pred_train = pd.DataFrame()
df_importance = pd.DataFrame()

print("load train dataset")

df_train = pd.read_feather("../../data/merge/train_merge.feather").iloc[random.sample(list(np.arange(590000)), 50000)]
print("load test dataset")
df_test = pd.read_feather("../../data/merge/test_merge.feather").iloc[random.sample(list(np.arange(500000)), 50000)]
df_train["isTrain"] = 1
df_test["isTrain"] = 0

df_all = pd.concat([df_train, df_test]).drop(remove_cols, axis=1, errors="ignore")
if select_cols is not None:
    df_all = df_all[list(select_cols) + [target_col] + [id_col]]

if is_reduce_memory:
    df_all = reduce_mem_usage(df_all)

pred_train, imp, result= learning(df_train=df_all)

df_pred_train = df_pred_train.append(pred_train, ignore_index=True)
imp.to_csv("{}/importance.csv".format(output_dir))

df_pred_train.to_csv("{}/predict_train.csv".format(output_dir), index=False)
result.to_csv("{}/result.csv".format(output_dir), index=False)