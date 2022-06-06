# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import re
import ast
import pandas as pd
import math
from typing import Dict, Set, List

from util import BinningMethod


# Predicate of the form attrib OP value
class Predicate:
    def __init__(self, attribute_name, operator, value):
        self.name = attribute_name
        self.op = operator
        self.val = value

    def _get_uid(self):
        return (self.name, self.op, self.val)

    def __str__(self):
        if self.op == '==' or self.op == '!=':
            value = '"{}"'.format(self.val)
        else:
            value = str(self.val)
        return self.name + self.op + value

    def dataframe_query(self):
        return self.__str__()

    @classmethod
    def from_string(cls, predicate_str: str):
        attribute_name, operator, value = re.split(r'(==|=|!=|<=|<|>=|>)', predicate_str)
        # partial order in predicate indicates numerical type
        if operator[0] == '<' or operator[0] == '>':
            value = ast.literal_eval(value)
        if operator == '=':
            operator = '=='
        return cls(attribute_name, operator, value)

    def num_positive(self, df: pd.DataFrame):
        #return sum(1 for _, row in data.iterrows() if self.eval(row))
        # or just call coverage from util
        return len(df.query(self.dataframe_query()))


    def eval(self, instance):
        inst_val = instance[self.name]
        if self.op == ">":
            return inst_val > self.val
        elif self.op == "<=":
            return inst_val <= self.val
        elif self.op == "==":
            return inst_val == self.val
        elif self.op == "!=":
            return inst_val != self.val

        raise Exception("attribute type not supported" + self.op)

    def __eq__(self, other):
        return self.name == other.name and self.op == other.op and self.val == other.val

    def __hash__(self):
        # We can use id(self) or proper hashing, this is not the issue I fixed (see other __hash__)
        return hash(self._get_uid())

    def implies(self, other):
        if self.name != other.name:
            return False

        if self.op == other.op and self.val == other.val:
            return True

        if self.op == ">":
            if other.op == ">" or other.op == "!=":
                return self.val > other.val
            else:
                return False

        if self.op == "<=":
            if other.op == "<=" or other.op == "!=":
                return self.val <= other.val
            else:
                return False

        if self.op == "==":
            if other.op == ">":
                return self.val > other.val
            if other.op == "<=":
                return self.val <= other.val
            return False

        if self.op == "!=":
            return False

        raise Exception("attribute type not supported" + self.op)


def filter(predicates: Set[Predicate], df: pd.DataFrame, pos: pd.DataFrame, predicate_relevance_threshold=0.01, minimum_relative_coverage=0):
    """
        This is a pre-processing step to identify a subset of interesting predicates
        to consider in the seed set.
        It considers a predicate to be "interesting" if the data filtered according
        to this predicate has a slightly higher occurrence of trait compared
        to whole population

        Parameters
        ----------

        predicates : Set[Predicate]
            The feature we care about (e.g., misprediction)

        df : pd.DataFrame
            Tabular data as Pandas data frame

        pos : pd.DataFrame
            Boolean value of target_attrib (e.g., for True, we want to predict when it's true)

        Returns
        -------
        filtered: Set[Predicate]
            A subset of filtered predicates
    """
    filtered = set()
    num_pos = pos.shape[0]
    num_total = df.shape[0]
    ratio_whole = num_pos / num_total

    for predicate in predicates:
        num_pass = predicate.num_positive(df)
        if num_pass == 0:
            continue
        # minimum_relative_coverage is given in %
        if num_pass / num_total < (minimum_relative_coverage/100):
            continue
        num_tp = predicate.num_positive(pos)
        ratio_current_predicate = num_tp / num_pass

        diff = ratio_current_predicate - ratio_whole

        if diff > predicate_relevance_threshold:
            filtered.add(predicate)
    return filtered


def generate(df: pd.DataFrame, relevant_attributes: Dict[str, str], num_bins: int, binning_method: BinningMethod) -> Set[Predicate]:
    """
        Generates a set of predicates based on the dataset and search hyper-parameters

        Parameters
        ----------

        predicates : Set[Predicate]
            The feature we care about (e.g., misprediction)

        df : pd.DataFrame
            Tabular data as Pandas data frame

        num_bins : int
            Number of bins

        binning_method : BinningMethod
            Either equal frequency or width binning

        Returns
        -------
        filtered: Set[Predicate]
            A subset of predicates
    """
    features: Set[Predicate] = set()
    for (attribute_name, attribute_type) in relevant_attributes.items():
        values = get_values_for_attributes(df, attribute_name)
        if attribute_type == 'D':
            features |= generate_discrete_predicates(attribute_name, values)
        elif attribute_type == 'C' or attribute_type == 'I':
            features |= generate_continous_predicates(attribute_name, values, num_bins, binning_method)
        else:
            raise Exception("attribute type not supported")
    return features


# Generates predicates for continuous values using discretization
# strategy specified by binning_method
def generate_continous_predicates(attribute_name: str, values: List, num_bins: int, binning_method: str) -> Set[Predicate]:
    if binning_method == BinningMethod.EQFreq:
        cutoffs = equi_freq(values, num_bins)
    elif binning_method == BinningMethod.EQWidth:
        cutoffs = equi_width(values, num_bins)

    preds = set()
    for val in cutoffs:
        pred_geq = Predicate(attribute_name, "<=", val)
        preds.add(pred_geq)
        pred_lt = Predicate(attribute_name, ">", val)
        preds.add(pred_lt)

    return preds


def generate_discrete_predicates(attribute_name: str, values: Set) -> Set[Predicate]:
    preds = set()
    # In a boolean attribute (e.g., True, False), there is no need to generate 4 predicates:
    # (attribute_name != values[0]) implies (attribute_name == values[1])
    if len(values) == 2:
        values = list(values)
        preds.add(Predicate(attribute_name, "==", values[0]))
        preds.add(Predicate(attribute_name, "==", values[1]))
        return preds

    for v in values:
        pred_eq = Predicate(attribute_name, "==", v)
        preds.add(pred_eq)
        pred_neq = Predicate(attribute_name, "!=", v)
        preds.add(pred_neq)
    return preds


# Returns cut-offs for discretization based on equal frequency binning
def equi_freq(values: Set, num_bins: int):
    cutoffs = set()
    sorted_values = sorted(values)

    num_values = len(sorted_values)

    # number of bins > number of unique values
    if num_bins > num_values:
        # Use square-root choice
        num_bins = math.ceil(math.sqrt(num_values))

    values_ratio = int(num_values / num_bins)

    for i in range(0, num_bins):
        arr = []
        for j in range(i * values_ratio, (i + 1) * values_ratio):
            if j >= num_values:
                break
            arr = arr + [sorted_values[j]]

        cutoff = arr[len(arr) - 1]
        cutoffs.add(cutoff)

    return cutoffs


# Returns cut-offs for discretization based on equal frequency binning
def equi_width(values: Set, num_bins: int):
    values = set(values)
    sorted_set = sorted(values)
    min_val = sorted_set[0]
    max_val = sorted_set[len(sorted_set) - 1]
    val_range = max_val - min_val

    cutoffs = []
    inc = val_range / num_bins
    inc = int(inc) + 1

    for i in range(min(num_bins, max_val)):
        cutoffs.append(i * inc)

    return cutoffs


def get_values_for_attributes(df: pd.DataFrame, attribute_name: str) -> Set:
    return set(df[attribute_name].unique())
