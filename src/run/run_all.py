# coding: utf-8

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
from src.run import remove_high_corr
from src.run import run
from src.run import run_timesplit
# from src.run import run_nn
from src.run import run_catboost
import glob

import time
import numpy as np

# print("waiting..D..alpha/lambda==1, colsample=0.025")
# time.sleep(60*60*1.5)
"""
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

f_999_merge.main(nrows=60000)
imp = run.main(params=None, experiment_name="lightgbm_all_removeagg", extract_feature=True)
imp = imp.sort_values("sum_importance", ascending=False)
"""

# test1
f_999_merge.main(nrows=None, drop_cols=imp["column"].iloc[1800:])
dummy = run.main(params=None, experiment_name="lightgbm_1800", extract_feature=False)
dummy = run.main(query="ProductCD == 'W'", params=None, experiment_name="lightgbm_1500_W", extract_feature=False)
dummy = run.main(query="ProductCD == 'C'", params=None, experiment_name="lightgbm_1500_C", extract_feature=False)
"""
# test2
w_imp = imp[~imp["column"].str.contains("uid")]
w_imp2 = imp[imp["column"].str.contains("uid")]
f_999_merge.main(nrows=None, drop_cols=np.concatenate([w_imp["column"].iloc[1800:].values, w_imp2["column"].values]))
dummy = run.main(params=None, experiment_name="lightgbm_all_remove_uid")
# catboostいろいろ

f_999_merge.main(nrows=None, drop_cols=imp["column"].iloc[1800:])
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
