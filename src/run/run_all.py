from src.feature2 import f_001_basic_feature
from src.feature2 import f_101_agg_id
from src.feature2 import f_102_countencoding
from src.feature2 import f_104_pattern
from src.feature2 import f_104_1_agg_id
from src.feature2 import f_105_pca
from src.feature2 import f_999_merge
from src.run import run
import time

print("all")
time.sleep(60*60*3)

print("basic_feature")
# f_001_basic_feature.main()
print("101_agg_id")
# f_101_agg_id.main()
print("102_countencoding")
# f_102_countencoding.main()
print("104_pattern")
# f_104_pattern.main()
print("104_1_agg_id")
# f_104_1_agg_id.main()
print("104_1_agg_id")
# f_105_pca.main()
print("999_merge")
f_999_merge.main()
print("run!")
run.main()
