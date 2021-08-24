#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, List
from collections.abc import Callable
from dataclasses import dataclass
import collections
import pandas as pd

from rules.ruleset import generate_ruleset, RuleSet
from rules.rule import Rule
from rules.conjuncts import generate_conjuncts
from util import sorted_pareto_optimal_rules, partition, ConfusionMatrix, BinningMethod, debug_list_log, bold, percent_format
import rules.predicate as Predicates
import cfg as cfg
import timing as Timing

"""
    Target 'class'
"""
Target = collections.namedtuple('Target', 'attribute value')

"""
    Configuration Settings
"""
@dataclass
class Settings:
    """
    beam_width : int
        Beam width controls the width of the beam during beam search.
        A large beam leads to more precision, but is slower

    num_rules : int
        Number of final rules to display

    num_bins : int
        The number of bins to use for discretization.

    binning_method : BinningMethod
        Binning method, currently only support EQFreq and EQWidth

    minimum_relative_coverage : float
        Determines the minimum relative amount of coverage (percentage of rows) each subgroup should have
        minimum_relative_coverage=1.5 means that each inferred subgroups has to have cover at least 1.5% of the total rows in the dataset

    disjunctions : bool
        Indicates whether the algorithm should also find disjunctions

    target_coverage : float
        The percentage of target that we want to cover (only applicable if Settings.disjunctions=True)
    """
    beam_width: int = 10
    num_rules: int = 3
    num_bins: int = 5
    binning_method: BinningMethod = BinningMethod.EQFreq
    minimum_relative_coverage: int = 0.1
    target_coverage: float = 0.6
    all_rules: bool = False
    disjunctions: bool = False
    debug_print_function: Callable = None


def discover(
    df: pd.DataFrame,
    target: Target,
    relevant_attributes: Dict[str, str],
    config: Settings = None
):
    """
    Disovers rules (conjunction of predicates over attributes) that succinctly
    describe a subgroup of a dataset with respect to a target attribute

    Given a tabular dataset, relevant features, and a target attribute,
    we compute a set of rules (conjunctions of predicates over feature values)
    that summarize how the target attribute can best be explained by the features
    (including coverage and precision values)

    Parameters
    ----------

    df : pd.DataFrame
        Tabular data as Pandas data frame

    target : NamedTuple[attribute: str, value: bool]
        attribute - The feature we care about (e.g., misprediction)
        value - Boolean value of target_attrib
        (e.g., for True, we want to predict when it's true)

    relevant_attributes : Dict[str, str]
        Mapping from relevant attributes to consider to their type (D (discrete), I (Int), C (Continuous))

    config : Settings
        Contains all configurable settings

    sample : pd.DataFrame
        A representative sample of the full dataset (optional)

    Returns
    -------
    Result containing final rules
        List of subgroups (conjunction of rules) describing the phenomenon of choice (target_attribute)
    """

    if config is None:
        config = Settings()

    # partition into positive and negative data subsets (wrt. target_attribute)
    (positive_df, negative_df) = partition(df, target)

    all_predicates = Predicates.generate(df, relevant_attributes, config.num_bins, config.binning_method)

    # print features before and after filtering
    debug_list_log("Predicates before filter: {}".format(len(all_predicates)), all_predicates, config.debug_print_function)

    # filter predicates that do not meet our thresholds
    relevant_predicates = Predicates.filter(all_predicates, df, positive_df, cfg.pred_relevance_threshold, config.minimum_relative_coverage)
    debug_list_log("Predicates after filter: {}".format(len(relevant_predicates)), relevant_predicates, config.debug_print_function)

    # generate beam_width number of rules
    rules = generate_conjuncts(df, target, relevant_predicates, config.beam_width)

    # return all rules if requested in the config
    if config.all_rules and not config.disjunctions:
        time_log = Timing.get_time_log()
        return Result(rules, df, target, time_log)

    # find the best num_rules that are Pareto optimal
    pareto_optimal_rules = sorted_pareto_optimal_rules(rules, df, target)
    best_rules = pareto_optimal_rules[:config.num_rules]
    debug_list_log("Best (pareto-optimal) rules: ", best_rules, config.debug_print_function)

    if config.disjunctions:
        final_rule_sets = []
        for r in best_rules:
            rs = generate_ruleset(r, df, target, relevant_predicates,
                                    config.target_coverage, config.beam_width)
            final_rule_sets.append(rs)
        return DisjunctiveResult(final_rule_sets, df, target)

    time_log = Timing.get_time_log()
    return Result(best_rules, df, target, time_log)


class ConfusionMatrixRuleSet(ConfusionMatrix):
    def coverage(self, ruleset, df=None):
        if df is None:
            df = self.df

        subgroups = [df.query(rule.dataframe_query()) for rule in ruleset.elems]
        subgroups_merged = pd.concat(subgroups, join='outer', axis=1).drop_duplicates()
        return subgroups_merged


class Result:
    def __init__(self, rules: List, df: pd.DataFrame, target: Target, time_log=None):
        self.rules = rules
        self.df = df
        self.target = target
        self.time_log = time_log

    def __repr__(self):
        final = []
        for r in self.rules:
            final.append(str(r))
        return "\n".join(final)

    """ Parses rule descriptions and turns them into a Result object

    Parameters
    ----------

    df : pd.DataFrame
        Tabular data as Pandas data frame

    rule_str : str
        A set of rules separated by newlines (\n)
        Each rule (on each line) is a conjunction of predicates (separated by '&')

    target: Target
        Target attribute and value

    Returns
    -------
    Result containing rules parsed from the string input

    """
    @classmethod
    def import_rules(cls, rule_str: str,  df: pd.DataFrame, target: Target):
        rules = list(map(Rule.from_string, rule_str.split('\n')))
        return cls(rules, df, target)

    def dataframe(self):
        result = []
        for rule in self.rules:
            result.append(self._rule_stats(rule, self.df, self.target))
        return pd.DataFrame(result).sort_values(by='precision', ascending=False)

    # computes precision, recall, coverage for each rule
    def _rule_stats(self, rule: Rule, df: pd.DataFrame, target):
        stats = ConfusionMatrix(df, target)
        rule_coverage = stats.num_covered(rule)
        return {
            'rule' : str(rule),
            'precision' : stats.get_tp_ratio(rule),
            'recall' : stats.get_coverage_ratio(rule),
            'rule_coverage' : rule_coverage/len(df),
        }

    def _print_rule(self, rule, df: pd.DataFrame, target):
        target_name, target_value = target
        rule_stat = self._rule_stats(rule, df, target)
        stats = ConfusionMatrix(df, target)

        print(bold("Subgroup: {}".format(str(rule))))

        overall_coverage = stats.num_covered(rule)

        print("% of subgroup in population (Full Dataset):\t{}% ({} rows)".format(
            percent_format(rule_stat['rule_coverage']),
            overall_coverage)
        )

        print("Precision: P({}={} | {}) = {}%".format(
            target_name,
            target_value,
            str(rule),
            percent_format(rule_stat['precision'])
        ))

        print("Recall: P({} | {}={}) = {}%".format(
            str(rule),
            target_name,
            target_value,
            percent_format(rule_stat['recall'])
        ))

    def print(self, df: pd.DataFrame = None):
        if df is None:
            df = self.df

        print("Subgroup Discovery Result\n")
        print("Found {} subgroups".format(bold(str(len(self.rules)))))

        print(bold("Dataset"))
        print("Target: {}={}".format(self.target[0], self.target[1]))
        print("# Rows:\t{}".format(df.shape[0]))
        print("# Cols:\t{}".format(df.shape[1]))

        pos, _ = partition(df, self.target)
        print("% Target in dataset {}%".format(percent_format(len(pos)/len(df))))

        for rule in self.rules:
            print("="*40)
            self._print_rule(rule, df, self.target)

    def get_rule_stats(self):
        return [
            self._rule_stats(rule, self.df, self.target)
            for rule in self.rules
        ]


class DisjunctiveResult:
    def __init__(self, rulesets: List[RuleSet], df: pd.DataFrame, target: Target):
        self.rulesets = rulesets
        self.df = df
        self.target = target

    # computes precision, recall, coverage for each rule
    def _rule_stats(self, ruleset: RuleSet, df: pd.DataFrame, target):
        stats = ConfusionMatrixRuleSet(df, target)
        rule_coverage = stats.num_covered(ruleset)
        return {
            'ruleset' : str(ruleset),
            'precision' : stats.get_tp_ratio(ruleset),
            'recall' : stats.get_coverage_ratio(ruleset),
            'rule_coverage' : rule_coverage/len(df)
        }

    def print(self):
        for ruleset in self.rulesets:
            print("#"*40)
            print(ruleset)
            print("#"*40)
            for rule in ruleset.elems:
                Result([rule], self.df, self.target).print()

    def dataframe(self):
        result = []
        for ruleset in self.rulesets:
            result.append(self._rule_stats(ruleset, self.df, self.target))
        return pd.DataFrame(result).sort_values(by='precision', ascending=False)
