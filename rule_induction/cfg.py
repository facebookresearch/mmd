# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
For selecting the seed set of parameters. This is the minimum positive deviation from
the base rate of occurrence of that feature in the population. Only then we consider it interesting.

In other words: We compute how often a predicate occurs within the target subgroup
and compare it to how often it occurs in the overall population.
Only when it occurs at a rate of [pred_relvance_threshold] more often in the target subgroup, we consider it further
"""
pred_relevance_threshold = 0.01

# same idea, applied to "true positive ratio"  of a rule for it to be interesting
rule_relevance_threshold = 0.05

# same for "coverage"
rule_coverage_threshold = 0.04


# Hyperparameters for rule scores (objective function)
coverage_conjunct = 1
size_conjunct = 5
tpr_conjunct = 7


max_size = 5  # used for calculating the size penalty.
MAX_ITERS = 5
min_tpr = 0.2

verbose = False
