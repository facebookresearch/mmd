import time
from typing import List
from .counterfactual_search import BaseCounterfactualSearch, GreedySearch
from .base_proxy import BasePerturbationProxy

class SequenceExplainer:
    def __init__(self, counterfactual_search: BaseCounterfactualSearch=None):
        self.counterfactual_search = counterfactual_search if counterfactual_search is not None \
                                        else GreedySearch(BasePerturbationProxy())


    def explain(self, document):
        start = time.perf_counter()
        sequence, full_explanations, perturbation_tracking = self.counterfactual_search.search(document)
        end = time.perf_counter()
        return SequenceExplanation(
                sequence,
                full_explanations,
                perturbation_tracking,
                execution_time=int(end-start),
                original_document=document
            )


class SequenceExplanation:
    def __init__(
        self,
        document_sequence: List,
        explanations: List,
        perturbation_tracking: List,
        execution_time: int=0,
        original_document=None
    ):
        self.document_sequence = document_sequence
        self.explanations = explanations
        self.perturbation_tracking = perturbation_tracking
        self.execution_time = execution_time
        self.original_document = original_document

    def has_explanations(self):
        return len(self.explanations) > 0

    # same as 'full' but without the positions
    def human_readable(self):
        return [
            list(map(lambda pos: self.document_sequence[pos], explanation_list[0]))
            for explanation_list in self.explanations
        ]

    def set_original_document(self, original_document):
        self.original_document = original_document

    def full(self):
        return [
            (
                list(
                    map(
                        lambda pos: (pos, self.document_sequence[pos]),
                        explanation_list[0],
                    )
                ),
                explanation_list[1],
            )
            for explanation_list in self.explanations
        ]

    # Returns a string representation as a list of explanations
    # Each explanation item is a tuple of document position and
    #item and document item at that position
    def __repr__(self):
        return str(self.full())

    def __str__(self):
        return str(self.human_readable())
