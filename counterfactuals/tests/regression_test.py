#!/usr/bin/env python3

import unittest
import sys
sys.path.append("..")

from typing import List, Tuple, Set

from counterfactuals.explainer import SequenceExplainer

from tests.test_proxies import RemoveWordsPerturbation
from counterfactuals.counterfactual_search import GreedySearch


class RegressionTest(unittest.TestCase):
    """
        Testing for regressions on a small (generated) dataset
    """
    def setUp(self):
        # instantiate mocked model and appropraite proxies
        self.proxy = RemoveWordsPerturbation()
        self.explainer = SequenceExplainer(GreedySearch(self.proxy))

    def test_empty_input(self):
        empty_input = ''
        expected_explanation_count = 0
        actual_explanation_count = len(self.explainer.explain(empty_input).explanations)
        self.assertEqual(actual_explanation_count, expected_explanation_count,
            msg='We should not be able to generate explanations for an empty input')

    def test_forbidden_word_input(self):
        forbidden_sentence = 'let us mix offensive apple and pear with forbidden pepper'

        # removing 'pepper' is the only explanation possible
        counterfactual_sentence = 'let us mix offensive apple and pear with forbidden'
        _, counterfactual_score = self.proxy.classify(counterfactual_sentence)

        # 'pepper' is the perturbed position 9
        expected_explanations = [([9], 0, counterfactual_score)]
        actual_explanations = self.explainer.explain(forbidden_sentence).explanations
        self.assertEqual(actual_explanations, expected_explanations)

        # also asserting that the explanation actually contains 'pepper'
        actual_human_readable_explanation = self.explainer.explain(forbidden_sentence).human_readable()
        expected_human_readable_explanation = [['pepper']]
        self.assertEqual(expected_human_readable_explanation, actual_human_readable_explanation)

    def test_offensive_forbidden_word_input(self):
        mix_sentence = 'let us mix offensive apple pear strawberry blueberry with forbidden eggplant'

        """
            Multiple scenarios of explanations are now possible, by removing:
            - Eggplant (forbidden word) + at least 1 offensive word
        """

        actual_explanations = list_to_set(self.explainer.explain(mix_sentence).human_readable())
        expected_explanations = {('eggplant', 'blueberry'), ('eggplant', 'strawberry'), ('eggplant', 'pear'), ('eggplant', 'apple')}
        self.assertEqual(expected_explanations, actual_explanations)


def list_to_set(lists: List) -> Set[Tuple[str]]:
    return {tuple(x) for x in lists}


if __name__ == '__main__':
    unittest.main()
