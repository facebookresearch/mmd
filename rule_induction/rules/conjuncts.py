# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import pandas as pd
import math
from timing import time_log

from .rule import Rule
from util import ConfusionMatrix


# Find a set of beam_width conjuncts (rules) based on our goodness criteria (using beam search)
# a Rule (aka conjunct) is a set of predicates conjoined by AND
# Note that objective function is implemented in Rule.eval
@time_log
def generate_conjuncts(df: pd.DataFrame, target, predicates, beam_width):
    stats = ConfusionMatrix(df, target)
    num_pos = stats.pos_df.shape[0]
    num_total = df.shape[0]
    min_support = math.sqrt(num_pos) / num_total

    #beam and new_beam are mappings from rules to q_values
    beam = {}
    new_beam = {}

    for pred in predicates:
        r = Rule()
        r.add_conjunct(pred)
        q = r.eval(stats)
        beam[r] = q

    if beam_width > len(predicates):
        new_beam = beam.copy()
    else:
        values = sorted(beam.values(), reverse=True)
        cutoff = values[beam_width - 1]
        new_beam = {key: value for (key, value) in beam.items() if value >= cutoff}
        beam = new_beam.copy()

    i = 0
    while True:
        improved = False
        for rule in list(beam):
            for pred in predicates:
                if rule.check_useless(pred):
                    continue

                new_rule = Rule(rule)
                new_rule.add_conjunct(pred)

                if is_duplicate(new_rule, new_beam):
                    continue

                q = new_rule.eval(stats)
                num_tp = len(stats.true_positive(new_rule))
                exceeds_min_support = num_tp / df.shape[0] >= min_support
                if not exceeds_min_support:
                    continue

                (worst_rule, worst_q) = get_worst_rule(new_beam, beam_width)

                if q <= worst_q:
                    continue

                # TP(new_rule) < TP(existing) AND
                # FP(new_rule) > FP(existing)
                if is_irrelevant(new_rule, new_beam, stats):
                    continue
                if worst_q > 0:
                    new_beam.pop(worst_rule)
                new_beam[new_rule] = q
                improved = True

        if i == 0:
            clean_beam(new_beam)

        if not improved:
            break

        beam = new_beam
        i += 1
    return beam


# is rule implied by existing rule in the beam?
def is_redundant(rule, new_beam):
    return any(r.implies(rule) for r in new_beam)


# is rule a duplicate of another rule in the beam?
def is_duplicate(rule, new_beam):
    return any(r == rule for r in new_beam)


# find worst rule in beam
def get_worst_rule(beam, beam_width):
    if len(beam) < beam_width:
        return (None, -1)
    worst_q = 2**60
    for cur_r in beam:
        cur_q = beam[cur_r]
        if cur_q < worst_q:
            worst_q = cur_q
            worst_rule = cur_r
    return (worst_rule, worst_q)


def clean_beam(beam):
    new_beam = {}
    for r in beam.keys():
        if len(r.elems) == 0:
            continue
        new_beam[r] = beam[r]
    return new_beam


def print_beam(beam, stats: ConfusionMatrix):
    sorted_beam = sorted(beam.items(), key=lambda x: x[1])
    for (rule, val) in sorted_beam:
        tpr = stats.get_tp_ratio(rule)
        coverage = stats.get_coverage_ratio(rule)
        print(rule, "q value: ", val, "\n", " True positive ratio ", tpr, " coverage: ", coverage)


# A rule R is irrelevant TP(R) < TP(existing) and FP(R) > FP(existing)
def is_irrelevant(rule: Rule, new_beam, stats: ConfusionMatrix):
    (new_q, new_tp, new_fp) = rule.get_eval_result(stats)
    for r in new_beam:
        if len(r.elems) == 0:
            continue
        (old_q, old_tp, old_fp) = r.get_eval_result(stats)
        if new_tp.issubset(old_tp) and old_fp.issubset(new_fp):
            return True
    return False
