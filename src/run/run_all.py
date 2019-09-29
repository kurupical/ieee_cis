# coding: utf-8

import pandas as pd
from src.feature2 import f_001_basic_feature
from src.feature2 import f_101_agg_id
from src.feature2 import f_101_agg_id_2
from src.feature2 import f_101_agg_id_3
from src.feature2 import f_102_countencoding
from src.feature2 import f_104_pattern
from src.feature2 import f_104_1_agg_id
from src.feature2 import f_105_pca
from src.feature2 import f_106_shift
from src.feature2 import f_107_rolling
from src.feature2 import f_108_pattern
# from src.feature2 import f_989_merge_nn
from src.feature2 import f_110_nan_pattern
from src.feature2 import f_999_merge
from src.run import remove_high_corr
from src.run import run
from src.run import run_timesplit
# from src.run import run_nn
from src.run import run_catboost
import glob
from sklearn.model_selection import KFold, StratifiedKFold, TimeSeriesSplit

import time
import numpy as np
# print("waiting.....")
# time.sleep(60*60*6)
"""
print("basic_feature")
f_001_basic_feature.main()
print("101_agg_id")
f_101_agg_id.main()
print("101_agg_id")
f_101_agg_id_2.main()
print("102_countencoding")
f_102_countencoding.main()
"""
"""
print("101_agg_id")
for i in range(10):
    print("\n\n\n\n\n *********** DIV = {} ************".format(i))
    f_101_agg_id_3.main(target_cols=["V{}".format(x) for x in range(1, 339 + 1) if x % 10 == i],
                        name="div{}".format(i))
print("104_pattern")
f_104_pattern.main()
print("104_1_agg_id")
f_104_1_agg_id.main()
print("105_1_agg_id")
f_105_pca.main()
print("106_shift")
# f_106_shift.main()
print("107_rolling")
# f_107_rolling.main()
f_108_pattern.main()
### SMALL TEST
params = {'num_leaves': 256,
          'min_child_samples': 200,
          'objective': 'binary',
          'max_depth': -1,
          'learning_rate': 0.05,
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

print("105_1_pca")
f_105_pca.main()

f_999_merge.main(nrows=60000)
dummy = run.main(params=params, experiment_name="lightgbm_ALL_PCA", extract_feature=False, is_reduce_memory=False, folds=KFold(5))
# test_agg
print("basic_feature")
f_001_basic_feature.main(is_debug=False)
print("101_agg_id")
f_101_agg_id.main()
merge_features = []
merge_features = glob.glob("../../data/101_agg_id/train/*.feather")
merge_features.extend(glob.glob("../../data/110_nan_pattern/train/*.feather"))
f_999_merge.main(nrows=200000, merge_features=merge_features)
dummy = run.main(params=params, experiment_name="lightgbm_onlyagg1_ID_nan", extract_feature=False, is_reduce_memory=False, folds=KFold(5))
"""
# f_001_basic_feature.main()
### MAIN
# f_999_merge.main(nrows=60000)
# imp = run.main(params=None, experiment_name="lightgbm_all", extract_feature=True, is_reduce_memory=False, folds=KFold(5))
# imp = imp.sort_values("sum_importance", ascending=False)
imp = pd.read_csv("../../output/20190924190129_lightgbm_all/importance.csv")
imp = imp.sort_values("sum_importance", ascending=False)

# test1
# f_999_merge.main(nrows=None, drop_cols=imp["column"].iloc[1200:])
"""
dummy = run.main(query="ProductCD == 'W'", params=None, experiment_name="lightgbm_1200_W", extract_feature=False, is_reduce_memory=False, folds=KFold(5))
dummy = run.main(query="ProductCD == 'C'", params=None, experiment_name="lightgbm_1200_C", extract_feature=False, is_reduce_memory=False, folds=KFold(5))
dummy = run.main(query="ProductCD == 'W' and D1 == 0", params=None, experiment_name="lightgbm_1200_W_D1is0", extract_feature=False, is_reduce_memory=False, folds=KFold(5))
dummy = run.main(query="DT_isDecember == 1", params=None, experiment_name="lightgbm_1200_isDecember", extract_feature=False, is_reduce_memory=False, folds=KFold(5))
dummy = run.main(query="DT_isDecember == 0", params=None, experiment_name="lightgbm_1200_isDecember", extract_feature=False, is_reduce_memory=False, folds=KFold(5))
dummy = run.main(params=None, experiment_name="lightgbm_1200_KFold", extract_feature=False, is_reduce_memory=False, folds=KFold(5))

# test2
w_imp = imp[~imp["column"].str.contains("\+DT")]
w_imp2 = imp[imp["column"].str.contains("\+DT")]
f_999_merge.main(nrows=None, drop_cols=np.concatenate([w_imp["column"].iloc[1200:].values, w_imp2["column"].values]))
dummy = run.main(params=None, experiment_name="lightgbm_all_remove_uid", is_reduce_memory=False, folds=KFold(5))
dummy = run.main(query="ProductCD == 'W'", params=None, experiment_name="lightgbm_1500_W_removeuid", extract_feature=False, is_reduce_memory=False, folds=KFold(5))
dummy = run.main(query="ProductCD == 'C'", params=None, experiment_name="lightgbm_1500_C_removeuid", extract_feature=False, is_reduce_memory=False, folds=KFold(5))
dummy = run.main(query="ProductCD == 'W' and D1 == 0", params=None, experiment_name="lightgbm_1500_W_D1is0_removeuid", extract_feature=False, is_reduce_memory=False, folds=KFold(5))

"""
params = {'num_leaves': 380,
          'min_child_samples': 100,
          'objective': 'binary',
          'max_depth': -1,
          'learning_rate': 0.01,
          "boosting_type": "gbdt",
          "subsample": 0.7,
          "bagging_seed": 11,
          "metric": 'auc',
          "verbosity": -1,
          'reg_alpha': 0.1,
          'reg_lambda': 0.1,
          'colsample_bytree': 0.05,
          'early_stopping_rounds': 200,
          'n_estimators': 20000,
          }
"""
dummy = run.main(params=params,
                 experiment_name="lightgbm_1200_KFold_timeseries",
                 extract_feature=False, is_reduce_memory=False, folds=TimeSeriesSplit(6))
dummy = run.main(query="ProductCD == 'W'", params=None, experiment_name="lightgbm_1200_W", extract_feature=False, is_reduce_memory=False, folds=KFold(5))
dummy = run.main(query="ProductCD == 'C'", params=None, experiment_name="lightgbm_1200_C", extract_feature=False, is_reduce_memory=False, folds=KFold(5))
# dummy = run.main(query="ProductCD == 'W' and D1 == 0", params=params, experiment_name="lightgbm_1200_W_D1is0", extract_feature=False, is_reduce_memory=False, folds=KFold(5))
# dummy = run.main(query="DT_isDecember == 1", params=params, experiment_name="lightgbm_1200_isDecember", extract_feature=False, is_reduce_memory=False, folds=KFold(5))
# dummy = run.main(query="DT_isDecember == 0", params=params, experiment_name="lightgbm_1200_isDecember", extract_feature=False, is_reduce_memory=False, folds=KFold(5))
"""

dummy = run.main(params=None, experiment_name="lightgbm_1200_KFold_timeseries", extract_feature=False, is_reduce_memory=False, folds=TimeSeriesSplit(6))
dummy = run.main(query="ProductCD == 'W'", params=None, experiment_name="lightgbm_1200_W_timeseries", extract_feature=False, is_reduce_memory=False, folds=TimeSeriesSplit(6))
dummy = run.main(query="ProductCD == 'C'", params=None, experiment_name="lightgbm_1200_C_timeseries", extract_feature=False, is_reduce_memory=False, folds=TimeSeriesSplit(6))
# dummy = run.main(query="ProductCD == 'W' and D1 == 0", params=params, experiment_name="lightgbm_1200_W_D1is0_timeseries", extract_feature=False, is_reduce_memory=False, folds=TimeSeriesSplit(6))
# dummy = run.main(query="DT_isDecember == 1", params=params, experiment_name="lightgbm_1200_isDecember_timeseries", extract_feature=False, is_reduce_memory=False, folds=TimeSeriesSplit(6))
# dummy = run.main(query="DT_isDecember == 0", params=params, experiment_name="lightgbm_1200_notDecember_timeseries", extract_feature=False, is_reduce_memory=False, folds=TimeSeriesSplit(6))

w_imp = imp[~imp["column"].str.contains("\+DT")]
w_imp2 = imp[imp["column"].str.contains("\+DT")]
f_999_merge.main(nrows=None, drop_cols=np.concatenate([w_imp["column"].iloc[1200:].values, w_imp2["column"].values]))
# dummy = run.main(params=None, experiment_name="lightgbm_all_remove_uid", is_reduce_memory=False, folds=KFold(5))
dummy = run.main(params=None, experiment_name="lightgbm_all_remove_uid_timeseries", is_reduce_memory=False, folds=TimeSeriesSplit(6))
"""
dummy = run.main(query="ProductCD == 'W'", params=None, experiment_name="lightgbm_1200_W_removeuid_timeseries", extract_feature=False, is_reduce_memory=False, folds=TimeSeriesSplit(6))
dummy = run.main(query="ProductCD == 'C'", params=None, experiment_name="lightgbm_1200_C_removeuid_timeseries", extract_feature=False, is_reduce_memory=False, folds=TimeSeriesSplit(6))
dummy = run.main(query="ProductCD == 'W' and D1 == 0", params=None, experiment_name="lightgbm_1200_W_D1is0_removeuid_timeseries", extract_feature=False, is_reduce_memory=False, folds=TimeSeriesSplit(6))
dummy = run.main(query="DT_isDecember == 1", params=None, experiment_name="lightgbm_1200_isDecember_timeseries", extract_feature=False, is_reduce_memory=False, folds=TimeSeriesSplit(6))
dummy = run.main(query="DT_isDecember == 0", params=None, experiment_name="lightgbm_1200_notDecember_timeseries", extract_feature=False, is_reduce_memory=False, folds=TimeSeriesSplit(6))
"""

f_999_merge.main(nrows=None, drop_cols=imp["column"].iloc[1200:])
"""
params = {
    'n_estimators': 12000,
    'learning_rate': 0.01,
    'eval_metric': 'AUC',
    'loss_function': 'Logloss',
    'random_seed': 0,
    'metric_period': 100,
    'od_wait': 200,
    'task_type': 'GPU',
    'max_depth': 11,
    "verbose": 100,
    "gpu_ram_part": 0.85
}
dummy = run.main(params=params, mode="catboost", experiment_name="catboost_1800_depth11", is_reduce_memory=False, folds=KFold(5))
"""
params = {
    'n_estimators': 12000,
    'learning_rate': 0.01,
    'eval_metric': 'AUC',
    'loss_function': 'Logloss',
    'random_seed': 0,
    'metric_period': 100,
    'od_wait': 200,
    'task_type': 'GPU',
    'max_depth': 11,
    "verbose": 100,
    "gpu_ram_part": 0.85
}
dummy = run.main(params=params, mode="catboost", experiment_name="catboost_1800_depth11", is_reduce_memory=False, folds=TimeSeriesSplit(6))