from typing import List, Tuple

class BasePerturbationProxy:
    initial_document: str

    def classify(self, document) -> Tuple[bool, float]:
        return NotImplementedError

    """
        Standard implementation of batch classify assumes
        a sequential `classify` function (i.e., batch size 1)
    """
    def batch_classify(self, document_list: List) -> List[Tuple[bool, float]]:
        if len(document_list) == 0:
            return []
        return list(map(self.classify, document_list))


    """
        Standard implementation assumes a set of words divided by spaces
        The perturbation space consists of all words in the document
    """
    def document_to_perturbation_space(self, document: str) -> List[str]:
        self.initial_document = document
        return document.split(' ')

    """
        Standard implementation simply removes tokens at certain positions
        in the perturbation space
    """
    def perturb_positions(self, perturbation_space: List, positions: List[int]) -> List:
        perturbed_sequence = []
        for i in range(len(perturbation_space)):
            if i not in positions:
                perturbed_sequence.append(perturbation_space[i])
        return ' '.join(perturbed_sequence)
