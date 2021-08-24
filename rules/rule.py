# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from .cache import cache, get_cache_key
from .predicate import Predicate
from util import ConfusionMatrix
import cfg as cfg


class Rule:
    # Rule : is a conjunction of predicates. We also call this a 'conjunct'
    def __init__(self, orig=None):
        self.elems = set(orig.elems) if orig else set()

    def add_conjunct(self, pred):
        self.elems.add(pred)

    def check_useless(self, pred):
        for p in self.elems:
            if pred.name != p.name:
                continue
            if p.op == ">":
                if pred.op == ">" or pred.op == "=" or pred.op == "!=":
                    return True
            if p.op == "<=":
                if pred.op == "<=" or pred.op == "=" or pred.op == "!=":
                    return True
            if p.op == "=":
                return True

            if p.op == "!=":
                return True

        return False

    # does this rule imply another one?
    def implies(self, other):
        for other_pred in other.elems:
            implied = False
            for self_pred in self.elems:
                if self_pred.implies(other_pred):
                    implied = True
                    break
            if not implied:
                return False

        return True

    def is_subset(self, other):
        for p in self.elems:
            contains_pred = any(p == q for q in other.elems)
            if not contains_pred:
                return False
        return True

    def __eq__(self, other):
        return self.is_subset(other) and other.is_subset(self)

    def __hash__(self):
        # Potential Issue: using id(self) is just comparison by pointer address
        # This results in failed matches during beam search
        # Using "hash(frozenset(self.elems))" below will produce a different (smaller) result
        #return id(self)
        return hash(frozenset(self.elems))

    def __str__(self):
        return ' & '.join(str(p) for p in self.elems)

    def __repr__(self):
        return self.__str__()

    def dataframe_query(self):
        return self.__str__()

    @classmethod
    def from_string(cls, rule_str : str):
        r = Rule()
        predicate_str_list = list(map(lambda x: x.strip(), rule_str.split('&')))
        for predicate_str in predicate_str_list:
            r.add_conjunct(Predicate.from_string(predicate_str))
        return r

    # Returns a triple (Q_value, TPs, FPs)
    def get_eval_result(self, stats):
        cache_key = get_cache_key(self, stats.df)
        if cache_key in cache.keys():
            (q, tps, fps, nc) = cache[cache_key]
            return (q, tps, fps)

        self.eval(stats)
        assert cache_key in cache.keys()
        (q, tps, fps, nc) = cache[cache_key]
        return (q, tps, fps)

    # Does this rule evaluate to true or false for this point?
    def eval_point(self, point):
        return self.covers(point)

    # Does this rule cover this point (i.e., does it evaluate to true?)
    def covers(self, point):
        return all(pred.eval(point) for pred in self.elems)

    def get_size(self):
        return len(self.elems)

    # Important function for calculating objective (q) value
    def eval(self, stats: ConfusionMatrix):
        cache_key = get_cache_key(self, stats.df)
        if cache_key in cache.keys():
            (q, tps, fps, nc) = cache[cache_key]
            return q

        tps = stats.true_positive(self)
        fps = stats.false_positive(self)
        nc = stats.num_covered(self)

        # Guide rule size with cfg.max_size parameter
        size_score = 1 - self.get_size() / cfg.max_size

        q = stats.get_coverage_ratio(self) * cfg.coverage_conjunct \
            + stats.get_tp_ratio(self) * cfg.tpr_conjunct \
            + size_score * cfg.size_conjunct

        cache[cache_key] = (q, tps, fps, nc)
        return q

    def is_better(self, other_rule, stats: ConfusionMatrix):
        """
            It's better if it's better both in terms of tpr and coverage
            This is used for filtering out rules that are not on the Pareto frontier
        """
        self_tpr = stats.get_tp_ratio(self)
        other_tpr = stats.get_tp_ratio(other_rule)
        self_coverage = stats.get_coverage_ratio(self)
        other_coverage = stats.get_coverage_ratio(other_rule)
        return (self_tpr > other_tpr and self_coverage > other_coverage)
