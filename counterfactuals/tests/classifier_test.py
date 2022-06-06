#!/usr/bin/env python3
from test_proxies import OffensiveSentenceClassifier

import unittest


class ClassiferTest(unittest.TestCase):
    """
        Testing (dummy) classifer provided for test
    """
    def setUp(self):
        self.classifier = OffensiveSentenceClassifier(
            # fruits are offensive
            offensive_words=['strawberry', 'blueberry', 'apple', 'pear', 'pineapple'],
            # vegetables are forbidden
            forbidden_words=['celery', 'pepper', 'carrot', 'zucchini', 'eggplant']
        )

    def test_empty_input(self):
        empty_input = ''
        expected_classifier_output = (0, 0)
        actual_classifier_output = self.classifier.predict(empty_input)
        self.assertEqual(actual_classifier_output, expected_classifier_output,
            msg='An empty sentence should have a negative label and 0 score')


    def test_classifier_scores(self):
        expected_outputs = {
            'let us try one offensive blueberry' :
            1 * self.classifier.OFFENSIVE_WORD_SCORE_INCREASE,

            'let us mix offensive apple ad pear with forbidden pepper' :
            2 * self.classifier.OFFENSIVE_WORD_SCORE_INCREASE + 0.5,
            # + (1-0)*self.classifier.FORBIDDEN_WORD_SCORE_INCREASE

            'let us go way over budget with vegetables celery pepper eggplant carrot' :
            0.5+(4-1)*self.classifier.FORBIDDEN_WORD_SCORE_INCREASE
        }
        for sentence, expected_score in expected_outputs.items():
            expected_label = 1 if expected_score >= 0.5 else 0
            expected_classifier_output = (expected_label, min(1.0, expected_score))
            actual_classifier_output = self.classifier.predict(sentence)
            self.assertEqual(actual_classifier_output, expected_classifier_output)

if __name__ == '__main__':
    unittest.main()
