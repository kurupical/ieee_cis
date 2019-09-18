import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
import optuna
from functools import partial
from datetime import datetime as dt
import os
import json
from glob import glob

def concat_data(df, dirs):
    for i, d in enumerate(dirs):
        w_df = pd.read_csv(d)
        w_df = w_df[w_df.columns[:2]]
        w_df.columns = ["TransactionID", "pred_{}".format(os.path.dirname(d))]
        df = pd.merge(df, w_df, on="TransactionID", how="left")
    return df

def objective(X, y, trial):
    weights = {}
    for col in X.columns:
        weights[col] = trial.suggest_uniform(col, 0, 1)
    return -score(X, y, weights)

def calc_weight(X, weights):
    weight_ary = []
    for col in X.columns:
        weight_ary.append(X[col].notnull().astype(int).values * weights[col])

    weight_ary = np.array(weight_ary).transpose(1, 0)
    # Normalization
    weight_ary = weight_ary / weight_ary.sum(axis=1).reshape(-1, 1)

    return (X.fillna(0).values * weight_ary).sum(axis=1)

def score(X, y, weights):

    pred = calc_weight(X, weights)
    return roc_auc_score(y, pred)


def main(data_train_dirs, data_submit_dirs=None, experiment_name="blending", maxmin_mode=False):
    """
    最適なweightを計算する！
    :param data_train_dirs:
    :param weights:
    :param data_submit_dirs:
    :return:
    """

    df = pd.read_csv("train_y.csv")         # train: TransactionID / y(isFraud)
    df = concat_data(df, data_train_dirs)   # df.columns = TransactionID, y, pred_1,2,3...
    if maxmin_mode:
        df["MAX"] = df.drop(["TransactionID", "y"], axis=1).max(axis=1)
        df["MIN"] = df.drop(["TransactionID", "y"], axis=1).min(axis=1)

    print(df.drop(["TransactionID", "y"], axis=1).corr().values)

    study = optuna.create_study()
    f = partial(objective, df.drop(["TransactionID", "y"], axis=1), df["y"].values)

    study.optimize(f, n_trials=1000)
    print("params: {}".format(study.best_params))

    df_result = pd.read_csv("../../data/original/sample_submission.csv")[["TransactionID"]]
    df_result = concat_data(df_result, data_submit_dirs) # df.columns = TransactionID, pred_1,2,3...
    if maxmin_mode:
        df_result["MAX"] = df_result.drop(["TransactionID"], axis=1).max(axis=1)
        df_result["MIN"] = df_result.drop(["TransactionID"], axis=1).min(axis=1)
    print(df_result.drop(["TransactionID"], axis=1).corr().values)
    df_result["isFraud"] = calc_weight(df_result.drop("TransactionID", axis=1), study.best_params)

    output_dir = "../../output/{}_{}".format(dt.now().strftime("%Y%m%d%H%M%S"), experiment_name)
    os.makedirs(output_dir)

    out_params = study.best_params
    out_params["score"] = study.best_value

    with open("{}/weights.json".format(output_dir), "w") as f:
        json.dump(study.best_params, f)
    df_result[["TransactionID", "isFraud"]].to_csv("{}/blending.csv".format(output_dir))


if __name__ == "__main__":
    """
    data_train_dirs = ["../../output/★20190915_lgbmALL+W+C/20190915163044/predict_train.csv",
                       "../../output/★20190915_lgbmALL+W+C/20190915174239/predict_train.csv",
                       "../../output/★20190915_lgbmALL+W+C/20190915175055/predict_train.csv"]
    data_submit_dirs = ["../../output/★20190915_lgbmALL+W+C/20190915163044/submit.csv",
                        "../../output/★20190915_lgbmALL+W+C/20190915174239/submit.csv",
                        "../../output/★20190915_lgbmALL+W+C/20190915175055/submit.csv"]
    data_train_dirs = ["../../output/CV0.9559_LB0.9557_(KFold)_20190912061746/predict_train.csv",
                       "../../output/★20190915_lgbmALL+W+C/20190915174239/predict_train.csv",
                       "../../output/★20190915_lgbmALL+W+C/20190915175055/predict_train.csv"]
    data_submit_dirs = ["../../output/CV0.9559_LB0.9557_(KFold)_20190912061746/submit.csv",
                        "../../output/★20190915_lgbmALL+W+C/20190915174239/submit.csv",
                        "../../output/★20190915_lgbmALL+W+C/20190915175055/submit.csv"]
    """

    data_train_dirs = glob("../../output/★20190927_lgbmALL/*/predict_train.csv")
    data_submit_dirs = glob("../../output/★20190927_lgbmALL/*/submit.csv")
    print(len(data_train_dirs))
    print(len(data_submit_dirs))

    main(data_train_dirs=data_train_dirs,
         data_submit_dirs=data_submit_dirs,
         experiment_name="blending_test")