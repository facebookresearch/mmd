import heapq
import logging
from .base_proxy import BasePerturbationProxy


class BaseCounterfactualSearch:
    def search(self, document, proxy: BasePerturbationProxy):
        raise NotImplementedError


class GreedySearch(BaseCounterfactualSearch):

    def __init__(self, proxy: BasePerturbationProxy, iterations: int=2):
        self.proxy = proxy
        self.iterations = iterations

    def search(self, document):
        output = self.proxy.classify(document)
        # accounting for both sequential and parallel implementation of classify
        initial_classification, initial_score = output[0] if isinstance(output, list) else output
        sequence = self.proxy.document_to_perturbation_space(document)
        exploration_candidates = []

        """
            In the first iteration, we can expand by every token in the sequence.
            As we find more explanations, this list decreases
        """
        possible_expansions = set(range(len(sequence)))

        explanations = []
        perturbation_tracking = []

        for i in range(self.iterations+1):
            best_candidate = choose_best_candidate(exploration_candidates)
            if not isinstance(best_candidate, list):
                best_candidate = [best_candidate]
            """
                Expand explanation size by 1 and order by likelihood
                of receiving to obtain a class change
            """
            candidates = [best_candidate + [expansion_word] for expansion_word in possible_expansions]
            counterfactual_documents = list(map(
                    lambda candidate_positions: self.proxy.perturb_positions(sequence, candidate_positions),
                    candidates
                ))
            logging.debug(f"(Candidates in Iteration {i}: {candidates}")

            logging.debug(f"Batching {len(counterfactual_documents)} forward passes")
            candidate_classifications = self.proxy.batch_classify(counterfactual_documents)


            for i in range(len(candidate_classifications)):
                classification, score = candidate_classifications[i]
                counterfactual_document = counterfactual_documents[i]
                candidate_positions = candidates[i]

                logging.debug(f"Document after perturbing positions {candidate_positions}: {counterfactual_document}")

                """
                    If the perturbation leads to a classification change
                    we have found a counterfactual explanation.

                    We add it to our list and also remove all positions
                    from the `possible_expansions` set
                """
                if initial_classification != classification:
                    """
                        Add to explanations set:
                            (1) The positions that were perturbed to achieve the classification change
                            (2) The new classification
                            (3) The new score
                    """
                    logging.debug(f"Adding the following position perturbations to explanations in iteration {i}: {candidate_positions}")
                    explanations.append((candidate_positions, classification, score))
                    # add new document to enable perturbation perturbation
                    perturbation_tracking.append(counterfactual_document)
                    # remove from possible_expansions
                    possible_expansions = possible_expansions - set(candidate_positions)
                else:
                    # add this to the exploration candidates
                    add_exploration_candidate(exploration_candidates, candidate_positions, score)
        return sequence, explanations, perturbation_tracking



"""
    Search util functions
"""


"""
    Guides the greedy heuristic by choosing
    candidates based on score differential
"""
def choose_best_candidate(exploration_candidates):
    if not exploration_candidates:
        return []
    score, candidate = heapq.heappop(exploration_candidates)
    return candidate

def add_exploration_candidate(exploration_candidates, candidate, score):
    """
        Adjust score to penalize longer explanation candidates
    """
    penalty = (len(candidate) - 1) * 0.1
    heapq.heappush(exploration_candidates, (score + penalty, candidate))
