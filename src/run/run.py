import pandas as pd
import lightgbm as lgb
import os
import numpy as np
from datetime import datetime as dt
from sklearn.model_selection import StratifiedKFold, KFold, TimeSeriesSplit
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
import copy
from tqdm import tqdm
import gc
import time
from src.feature.common import reduce_mem_usage
from catboost import  CatBoostClassifier
import json
from scipy.stats import ks_2samp

# hyper parameters
n_folds = 5
random_state = 0
n_loop = 1
target_col = "isFraud"
id_col = "TransactionID"
remove_cols = ["TransactionDT",
               "id_13", "id_30", "id_31"# train/testで分布が違いすぎる (別手法で対応),
               # "card1", "card2", "card3", "card5",
               # "addr1", "addr2"
               ]
remove_cols.extend(pd.read_csv("cols.csv")["column"].values)

is_reduce_memory = True
select_cols = None # 全てのcolumnを選ぶ
# select_cols = pd.read_csv("cols.csv")["column"].values

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

def learning_lgbm(df_train, df_test, params, extract_feature, folds):

    if params is None:
        params = {'num_leaves': 256,
                  'min_child_samples': 200,
                  'objective': 'binary',
                  'max_depth': -1,
                  'learning_rate': 0.01,
                  "boosting_type": "gbdt",
                  "subsample": 0.7,
                  "bagging_seed": 11,
                  "metric": 'auc',
                  "verbosity": -1,
                  'reg_alpha': 1,
                  'reg_lambda': 1,
                  'colsample_bytree': 0.05,
                  'early_stopping_rounds': 200,
                  'n_estimators': 20000,
                  }

    cat_feats = _get_categorical_features(df_test)

    for f in cat_feats:
        # print(f)
        df_train[f] = df_train[f].fillna("nan").astype(str)
        df_test[f] = df_test[f].fillna("nan").astype(str)
        le = LabelEncoder().fit(df_train[f].append(df_test[f]))
        df_train[f] = le.transform(df_train[f])
        df_test[f] = le.transform(df_test[f])

    i = 0
    """
    if extract_feature:
        folds = KFold(n_splits=n_folds)
    else:
        folds = KFold(n_splits=n_folds)
    """

    print("-----------------------------------")
    print("LOOP {} / {}".format(i+1, n_loop))
    print("-----------------------------------")

    df_pred_train = pd.DataFrame()
    df_pred_test = pd.DataFrame()
    print(df_train.shape)
    print(df_test.shape)
    df_pred_test[id_col] = df_test[id_col]
    df_importance = pd.DataFrame()
    df_result = pd.DataFrame()
    df_importance["column"] = df_train.drop([id_col, target_col], axis=1).columns
    final_iteration = 0

    idxs = []
    for idx in folds.split(df_train, df_train[target_col]):
        idxs.append(idx)
    if folds.__class__ == TimeSeriesSplit:
        idxs.append((np.arange(len(df_train)), np.arange(len(df_train))[-100:]))

    for n_fold, (train_idx, val_idx) in enumerate(idxs):
        print("-----------------------------------")
        print("Fold {} / {}".format(n_fold+1, n_folds))
        print("-----------------------------------")
        # X = df_train.drop([id_col, target_col, *remove_cols], axis=1)
        # y = df_train[target_col]
        # if n_fold == 0: continue
        # if n_fold == 1: continue
        # if n_fold == 2: continue
        # if n_fold == 4: continue
        print(train_idx)
        print(val_idx)

        if extract_feature:
            if n_fold != 4:
                continue

        if folds.__class__ == TimeSeriesSplit:
            if n_fold < 3:
                continue
            if n_fold == 5:
                params["n_estimators"] = final_iteration * 1.1
                params["early_stopping_rounds"] = 10000  # disable

        lgb_train = lgb.Dataset(data=df_train.drop([id_col, target_col], axis=1).iloc[train_idx], label=df_train[target_col].iloc[train_idx])
        lgb_val = lgb.Dataset(data=df_train.drop([id_col, target_col], axis=1).iloc[val_idx], label=df_train[target_col].iloc[val_idx])

        model = lgb.train(copy.copy(params),
                          lgb_train,
                          categorical_feature=cat_feats,
                          valid_sets=[lgb_train, lgb_val],
                          verbose_eval=100)
        final_iteration = model.current_iteration()
        w_pred_train = model.predict(df_train.drop([id_col, target_col], axis=1).iloc[val_idx])
        df_pred_train = df_pred_train.append(pd.DataFrame(
            {id_col: df_train[id_col].iloc[val_idx],
             "pred": w_pred_train,
             "y": df_train[target_col].iloc[val_idx]}
        ), ignore_index=True)
        w_pred_test = model.predict(df_test.drop([id_col], axis=1))
        print(w_pred_test.mean())
        df_pred_test["pred_fold{}_{}".format(i, n_fold)] = w_pred_test
        w_df_importance = pd.DataFrame()
        w_df_importance["column"] = df_train.drop([id_col, target_col], axis=1).columns
        w_df_importance["fold{}_{}_gain".format(i, n_fold)] = \
            model.feature_importance(importance_type="gain") / model.feature_importance(importance_type="gain").sum()
        w_df_importance["fold{}_{}_split".format(i, n_fold)] = \
            model.feature_importance(importance_type="split") / model.feature_importance(importance_type="split").sum()
        df_importance = pd.merge(df_importance, w_df_importance, how="left", on="column").fillna(0)
        df_result = df_result.append(
            pd.DataFrame(
                {"fold": [n_fold],
                 "random_state": [random_state],
                 "auc_train": [roc_auc_score(df_train[target_col].iloc[train_idx], model.predict(df_train.drop([id_col, target_col], axis=1).iloc[train_idx]))],
                 "auc_test": [roc_auc_score(df_train[target_col].iloc[val_idx], w_pred_train)]}
            ),
            ignore_index=True
        )
        del w_pred_train, w_pred_test, lgb_train, lgb_val, model
        gc.collect()

    df_submit = pd.DataFrame()
    df_submit[id_col] = df_test[id_col]
    if folds.__class__ == TimeSeriesSplit:
        weights = [0.05, 0.15, 0.3, 0.5]
        df_submit[target_col] = df_pred_test.drop(id_col, axis=1).mul(weights).sum(axis=1)
    else:
        df_submit[target_col] = df_pred_test.drop(id_col, axis=1).mean(axis=1)
    return df_submit, df_pred_train, df_pred_test, df_importance, df_result


def learning_catboost(df_train, df_test, params, extract_feature, folds):
    if params is None:
        params = {
            'n_estimators': 12000,
            'learning_rate': 0.03,
            'eval_metric': 'AUC',
            'loss_function': 'Logloss',
            'random_seed': random_state,
            'metric_period': 100,
            'od_wait': 200,
            'task_type': 'GPU',
            'max_depth': 8,
            "verbose": 100
        }
    cat_feats = _get_categorical_features(df_test)


    for f in cat_feats:
        # print(f)
        df_train[f] = df_train[f].astype(str)
        df_test[f] = df_test[f].astype(str)

    i = 0
    # folds = KFold(n_splits=n_folds, shuffle=False)

    print("-----------------------------------")
    print("LOOP {} / {}".format(i+1, n_loop))
    print("-----------------------------------")

    df_pred_train = pd.DataFrame()
    df_pred_test = pd.DataFrame()
    print(df_test)
    df_pred_test[id_col] = df_test[id_col]
    df_importance = pd.DataFrame()
    df_result = pd.DataFrame()
    df_importance["column"] = df_train.drop([id_col, target_col], axis=1).columns
    for n_fold, (train_idx, val_idx) in enumerate(folds.split(df_train, y=df_train[target_col])):
        print("-----------------------------------")
        print("Fold {} / {}".format(n_fold+1, n_folds))
        print("-----------------------------------")
        # X = df_train.drop([id_col, target_col, *remove_cols], axis=1)
        # y = df_train[target_col]
        # if n_fold == 0: continue
        # if n_fold == 1: continue
        # if n_fold == 2: continue
        # if n_fold == 4: continue

        model = CatBoostClassifier(**params)

        model.fit(df_train.drop([id_col, target_col], axis=1).iloc[train_idx],
                  df_train[target_col].iloc[train_idx],
                  cat_features=cat_feats,
                  eval_set=(df_train.drop([id_col, target_col], axis=1).iloc[val_idx], df_train[target_col].iloc[val_idx]))

        w_pred_train = model.predict_proba(df_train.drop([id_col, target_col], axis=1).iloc[val_idx])[:, 1]
        df_pred_train = df_pred_train.append(pd.DataFrame(
            {id_col: df_train[id_col].iloc[val_idx],
             "pred": w_pred_train,
             "y": df_train[target_col].iloc[val_idx]}
        ), ignore_index=True)
        w_pred_test = model.predict_proba(df_test.drop([id_col], axis=1))[:, 1]
        print(w_pred_test.mean())
        df_pred_test["pred_fold{}_{}".format(i, n_fold)] = w_pred_test
        """
        df_importance["fold{}_{}_gain".format(i, n_fold)] = \
            model.feature_importance(importance_type="gain") / model.feature_importance(importance_type="gain").sum()
        df_importance["fold{}_{}_split".format(i, n_fold)] = \
            model.feature_importance(importance_type="split") / model.feature_importance(importance_type="split").sum()
        """
        df_result = df_result.append(
            pd.DataFrame(
                {"fold": [n_fold],
                 "random_state": [random_state],
                 "auc_train": [roc_auc_score(df_train[target_col].iloc[train_idx], model.predict_proba(df_train.drop([id_col, target_col], axis=1).iloc[train_idx])[:, 1])],
                 "auc_test": [roc_auc_score(df_train[target_col].iloc[val_idx], w_pred_train)]}
            ),
            ignore_index=True
        )
        del w_pred_train, w_pred_test, model
        gc.collect()

        if extract_feature:
            break

    df_submit = pd.DataFrame()
    df_submit[id_col] = df_test[id_col]
    df_submit[target_col] = df_pred_test.drop(id_col, axis=1).mean(axis=1)
    return df_submit, df_pred_train, df_pred_test, df_importance, df_result


def main(query=None, params=None, experiment_name="", mode="lightgbm", extract_feature=False, is_reduce_memory=False,
         folds=None):
    # print("waiting...")
    # time.sleep(60*60*0.5)
    output_dir = "../../output/{}_{}".format(dt.now().strftime("%Y%m%d%H%M%S"), experiment_name)
    os.makedirs(output_dir)

    if params is not None:
        with open("{}/param.json".format(output_dir), "w") as f:
            json.dump(params, f)


    df_submit = pd.DataFrame()
    df_pred_train = pd.DataFrame()
    df_pred_test = pd.DataFrame()
    df_importance = pd.DataFrame()

    print("load train dataset")
    df_train = pd.read_feather("../../data/merge/train_merge.feather").drop(remove_cols, axis=1, errors="ignore")
    print("load test dataset")
    df_test = pd.read_feather("../../data/merge/test_merge.feather").drop(remove_cols, axis=1, errors="ignore")

    allnull_test = df_test.isnull().sum()
    allnull_test = allnull_test[allnull_test == len(df_test)].index

    print(allnull_test)
    df_train = df_train.drop(allnull_test, axis=1)
    df_test = df_test.drop(allnull_test, axis=1)
    if select_cols is not None:
        df_train = df_train[list(select_cols) + [target_col] + [id_col]]
        df_test = df_test[list(select_cols) + [id_col]]

    if query is not None:
        df_train = df_train.query(query).reset_index(drop=True)
        df_test = df_test.query(query).reset_index(drop=True)
    if extract_feature:
        is_reduce_memory = False

    if is_reduce_memory:
        df_train = reduce_mem_usage(df_train)
        df_test = reduce_mem_usage(df_test)

    if mode == "lightgbm":
        sub, pred_train, pred_test, imp, result = learning_lgbm(df_train=df_train,
                                                                df_test=df_test,
                                                                params=params,
                                                                extract_feature=extract_feature,
                                                                folds=folds)

    if mode == "catboost":
        sub, pred_train, pred_test, imp, result = learning_catboost(df_train=df_train,
                                                                    df_test=df_test,
                                                                    params=params,
                                                                    extract_feature=extract_feature,
                                                                    folds=folds)

    df_submit = df_submit.append(sub, ignore_index=True)
    df_pred_train = df_pred_train.append(pred_train, ignore_index=True)
    df_pred_test = df_pred_test.append(pred_test, ignore_index=True)
    imp["sum_importance"] = imp.sum(axis=1)
    imp.to_csv("{}/importance.csv".format(output_dir))

    df_submit.to_csv("{}/submit.csv".format(output_dir), index=False)
    df_pred_train.to_csv("{}/predict_train.csv".format(output_dir), index=False)
    df_pred_test.to_csv("{}/submit_detail.csv".format(output_dir), index=False)
    result.to_csv("{}/result.csv".format(output_dir), index=False)

    return imp


if __name__ == "__main__":
    main()