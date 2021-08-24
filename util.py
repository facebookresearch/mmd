# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import pandas as pd
import logging
from enum import Enum
from typing import List
from collections.abc import Callable

from rules.cache import cache, get_cache_key
from timing import time_log


def partition(
    df: pd.DataFrame,
    target
):
    """ Partitions data into positive and negative subsets

        Parameters
        ----------

        df : pd.DataFrame
            Tabular data as Pandas data frame

        target_attribute : str
            The feature we care about (e.g., misprediction)

        target_value : bool
            Boolean value of target_attrib (e.g., for True, we want to predict when it's true)

        Returns
        -------
        (pos_data, neg_data): Tuple[pd.DataFrame, pd.DataFrame]
            Tuple of positive and negative subsets of data in df wrt. target attribute and value

    """
    target_name, target_value = target
    positive_df = df[df[target_name] == target_value]
    negative_df = df[df[target_name] != target_value]
    return (positive_df, negative_df)


class ConfusionMatrix:
    def __init__(self, df: pd.DataFrame, target):
        self.df = df
        self.pos_df, self.neg_df = partition(df, target)

    def true_positive(self, rule):
        cache_key = get_cache_key(rule, self.df)
        if cache_key in cache.keys():
            (q, tps, fps, nc) = cache[cache_key]
            return tps
        tp_coverage = self.coverage(rule, self.pos_df)
        return {u_id for u_id, row in tp_coverage.iterrows()}

    def false_positive(self, rule):
        cache_key = get_cache_key(rule, self.df)
        if cache_key in cache.keys():
            (q, tps, fps, nc) = cache[cache_key]
            return fps
        fp_coverage = self.coverage(rule, self.neg_df)
        return {u_id for u_id, row in fp_coverage.iterrows()}

    # returns the number of data points covered by rule
    def num_covered(self, rule):
        cache_key = get_cache_key(rule, self.df)
        if cache_key in cache.keys():
            (q, tps, fps, nc) = cache[cache_key]
            return nc
        return len(self.coverage(rule))

    def coverage(self, rule, df=None):
        if df is None:
            df = self.df
        return df.query(rule.dataframe_query())

    # P(feature | misprediction) - recall
    def get_coverage_ratio(self, rule):
        num_tp = len(self.true_positive(rule))
        pos_length = self.pos_df.shape[0]
        return num_tp / pos_length if pos_length > 0 else 0

    # P(misprediction | feature) - precision
    def get_tp_ratio(self, rule):
        num_tp = len(self.true_positive(rule))
        nc = self.num_covered(rule)
        if nc == 0:
            return 0
        return num_tp / nc


# Returns the best rule according to q value after
# filtering out non-pareto-optimal rules
@time_log
def sorted_pareto_optimal_rules(rules, df: pd.DataFrame, target):
    # filters out rules that are not on the Pareto frontier
    def retain_pareto_optimal(rules: List, stats: ConfusionMatrix):
        retained = {}
        filtered = set()
        for r1 in rules.keys():
            keep = True
            for r2 in rules.keys():
                if r1 == r2:
                    continue
                if r2 in filtered:
                    continue
                # there is some rule that is better than r1,
                # so we don't need to keep r1
                if r2.is_better(r1, stats):
                    keep = False
                    filtered.add(r1)
                    break
            if keep:
                retained[r1] = rules[r1]
        return retained

    stats = ConfusionMatrix(df, target)
    pareto_optimal = retain_pareto_optimal(rules, stats)
    sorted_rules = sorted(pareto_optimal.items(), key=lambda x: x[1], reverse=True)
    # leave out score
    return [rule for rule, _ in sorted_rules]


class BinningMethod(Enum):
    EQFreq = 1
    EQWidth = 2

def debug_list_log(msg, list=None, log_method:Callable = None):
    if log_method is None:
        log_method = logging.debug

    log_method(msg)
    if list is not None:
        for list_item in list:
            log_method(list_item)

def bold(text: str):
    return '\033[1m' + text + '\033[0m'

def percent_format(percent):
    assert(percent >= 0 and percent <= 1)
    return round(percent*100, 2)
