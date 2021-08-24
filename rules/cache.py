# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from pandas.util import hash_pandas_object
import pandas as pd

# Mapping from (rule_str, dataframe_hash) pairs to a tuple (q, tps, fps, num_covered)
cache = {}

# Given a rule and data frame, returns the
# cache key used for memoization
def get_cache_key(rule, df: pd.DataFrame):
    h = hash_pandas_object(df)
    cache_key = (str(rule) + ":" + str(h))
    return cache_key
