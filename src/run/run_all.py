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
from src.feature2 import f_999_merge
from src.run import run
from src.run import run_timesplit
# from src.run import run_nn
from src.run import run_catboost

import time
"""
# print("waiting..D..alpha/lambda==1, colsample=0.025")
# time.sleep(60*60*1.5)
print("basic_feature")
f_001_basic_feature.main()
print("101_agg_id")
f_101_agg_id.main()
print("101_agg_id")
f_101_agg_id_2.main()
print("101_agg_id")
for i in range(10):
    print("\n\n\n\n\n *********** DIV = {} ************".format(i))
    f_101_agg_id_3.main(target_cols=["V{}".format(x) for x in range(1, 339 + 1) if x % 10 == i],
                        name="div{}".format(i))
print("102_countencoding")
f_102_countencoding.main()
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
"""
print("999_merge")
f_999_merge.main(nrows=65000)
print("run!")
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
          'n_estimators': 20000
          }
run.main(params=params, experiment_name="lightgbm_allfeats_108pattern")
print("run!")
# catboostいろいろ

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
    "verbose": 100
}
run_catboost.main(params)

params = {
    'n_estimators': 12000,
    'learning_rate': 0.01,
    'eval_metric': 'AUC',
    'loss_function': 'Logloss',
    'random_seed': 0,
    'metric_period': 100,
    'od_wait': 200,
    'task_type': 'GPU',
    'max_depth': 12,
    "verbose": 100
}
run_catboost.main(params)

#
# print("run!")
# run_timesplit.main()
