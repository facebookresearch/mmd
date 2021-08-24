# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import pandas as pd
from typing import Set
import math

from .rule import Rule
from .predicate import Predicate
from .conjuncts import generate_conjuncts
from util import sorted_pareto_optimal_rules, ConfusionMatrix
import cfg as cfg


def generate_ruleset(
    rule: Rule,
    df: pd.DataFrame,
    target,
    predicates: Set[Predicate],
    target_coverage: float,
    beam_width: int
):
    """
        Discovers a set of rules until target_coverage is achieved or until a certain number of
        rules are generated. The idea is to find one rule, then filter out part of the
        data set covered by that rule, and then find the next rule etc. until
        termination condition holds (e.g., target coverage is achieved)

        rule : Rule
            Initial rule seed

        df : pd.DataFrame
            Full dataset

        target : Target
            Target variable (attribute_name, attribute_value)

        predicates : Set[Predicate]
            Relevant predicates

        target_coverage : float
            The percentage of target_attrib that we want to cover

        beam_width : int
            Beam width controls the width of the beam during beam search.
            A large beam leads to more precision, but is slower

        Returns
        -------
        rs: RuleSet
            Mined ruleset

    """
    rs = RuleSet()
    rs.add_rule(rule)  # start with the rule that was passed in

    cur_df = df
    stats = ConfusionMatrix(df, target)
    cur_pos = stats.pos_df
    cur_neg = stats.neg_df

    for _ in range(cfg.MAX_ITERS):
        # remove the points for which the 'rule' was true
        cur_df = filter_by_rule(cur_df, rule)
        cur_pos = filter_by_rule(cur_pos, rule)
        cur_neg = filter_by_rule(cur_neg, rule)

        rules = generate_conjuncts(cur_df, target, predicates, beam_width)
        if len(rules) == 0:
            break

        rules = filter_rules(rules, stats)
        if len(rules) == 0:
            break
        best_rules = sorted_pareto_optimal_rules(rules, df, target)
        rule = best_rules[0]  # pick the very top one

        if len(rule.elems) == 0:
            break

        rs.add_rule(rule)

        # Coverage of a disjunctive ruleset is the coverage sum of all conjunctions
        cur_coverage = sum(stats.get_coverage_ratio(rule) for rule in rs.elems)
        if cur_coverage >= target_coverage:
            break

    return rs


# Filters out parts of data that are covered by the given rule
def filter_by_rule(df: pd.DataFrame, rule: Rule):
    new_data = pd.DataFrame.copy(df)
    indices = [i for i, (_, row) in enumerate(df.iterrows()) if rule.eval_point(row)]
    new_data.drop(new_data.index[indices], inplace=True)
    return new_data


# performs statistical significant test to identify interesting rules
def filter_rules(rules, stats):
    filtered = {}
    for r in rules.keys():
        if pass_threshold(r, stats):
            filtered[r] = rules[r]
    return filtered

# checks whether the given rule or rule set
# passes various thresholds in terms
# of true positive ratio and coverage
def pass_threshold(r, stats: ConfusionMatrix):
    num_pos = stats.pos_df.shape[0]
    num_total = stats.df.shape[0]
    ratio_whole = num_pos / num_total
    min_support = math.sqrt(num_pos) / num_total

    tps = stats.true_positive(r)
    num_tp = len(tps)
    exceeds_min_support = num_tp / stats.df.shape[0] >= min_support
    if not exceeds_min_support:
        return False
    cur_tpr = stats.get_tp_ratio(r)
    diff = cur_tpr - ratio_whole
    if diff < cfg.rule_relevance_threshold:
        return False
    cur_coverage = stats.get_coverage_ratio(r)
    return cur_coverage > cfg.rule_coverage_threshold


class RuleSet:
    """
        Ruleset : is a set of Rules, implicitly connected by a disjunction.
    """
    def __init__(self, orig=None):
        self.elems = set(orig.elems) if orig else set()

    def add_rule(self, rule):
        self.elems.add(rule)

    def __str__(self):
        return ' | '.join(f'({p})' for p in self.elems)

    def __eq__(self, other):
        """Overrides the default implementation"""
        return self.equals(other)

    def __hash__(self):
        return hash(frozenset(self.elems))

    # Does this ruleSet cover this point (i.e., does it evaluate to true?)
    def covers(self, point):
        return any(rule.eval_point(point) for rule in self.elems)

    # Does this rule evaluate to true or false for this point?
    def eval_point(self, point):
        return self.covers(point)

    def is_subset(self, other):
        for r in self.elems:
            contains_rule = any(r == x for x in other.elems)
            if not contains_rule:
                return False
        return True

    def equals(self, other):
        return self.is_subset(other) and other.is_subset(self)

    def impliesRule(self, rule):
        return any(r.implies(rule) for r in self.elems)

    def impliesRuleSet(self, other):
        return all(self.impliesRule(r) for r in other.elems)

    def check_useless(self, rule):
        if self.impliesRule(rule):
            return True
        rs = RuleSet()
        rs.add_rule(rule)
        return rs.impliesRuleSet(self)

    def get_size(self):
        return sum(r.get_size() for r in self.elems)
