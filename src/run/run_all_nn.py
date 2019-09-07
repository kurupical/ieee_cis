
from src.feature2 import f_989_merge_nn
from src.run import run_nn

import time

print("waiting..")
time.sleep(60*60*5)

print("989_merge_nn")
f_989_merge_nn.main()
# print("run_nn")
# run_nn.main(model_name="basic")
print("run_nn")
run_nn.main(model_name="basic_deep")
print("run_nn")
run_nn.main(model_name="basic_deep2")
# print("run_nn")
# run_nn.main(model_name="basic_deep2_ce")

