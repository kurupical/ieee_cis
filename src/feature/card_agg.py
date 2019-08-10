import pandas as pd
import numpy as np

def agg(df):

    target_cols = ["card1", "card2"]
    agg_cols = ["C{}".format(x) for i in range(1, 14)]
    agg_cols.append(["V{}".format(x) for i in range(1, 339+1)])

    for t_col in target_cols:
        for a_col in agg_cols:
            df["{}div{}".format(a_col, t_col)] = 