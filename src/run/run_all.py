from src.feature2 import f_001_basic_feature
from src.feature2 import f_101_agg_id
from src.feature2 import f_101_agg_id_2
from src.feature2 import f_101_agg_id_3
from src.feature2 import f_102_countencoding
from src.feature2 import f_104_pattern
from src.feature2 import f_104_1_agg_id
from src.feature2 import f_105_pca
# from src.feature2 import f_989_merge_nn
from src.feature2 import f_999_merge
from src.run import run
from src.run import run_timesplit
# from src.run import run_nn

import time

print("waiting..")
# time.sleep(60*60*1)
print("basic_feature")
# f_001_basic_feature.main()

# print("101_agg_id")
# f_101_agg_id.main()
# print("101_agg_id")
# f_101_agg_id_2.main()
"""
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
"""

print("999_merge")
f_999_merge.main(nrows=80000)
print("run!")
# run.main()
print("run!")
run_timesplit.main()
