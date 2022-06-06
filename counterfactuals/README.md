# Counterfactual Explanatins for Models of Code

Counterfactual explanations ([counterfactuals](/counterfactuals)) constitute minimal changes to the input space under which the model
“changes its mind". The contrast between original input and perturbed input is considered an explanation.

This work has been published at the International Conference on Software Engineering (ICSE'22), Software Engineering in Practice: J. Cito, I. Dillig, V. Murali, S. Chandra, [Counterfactual Explanations for Models of Code](https://arxiv.org/pdf/2111.05711.pdf).

```bibtex
@inproceedings{code_counterfactuals:22,
  title={Counterfactual Explanations for Models of Code},
  author={Cito, J{\"u}rgen and Dillig, Isil and Murali, Vijayaraghavan and Chandra, Satish},
booktitle = {44th {IEEE/ACM} International Conference on Software Engineering:
               Software Engineering in Practice, {ICSE} {(SEIP)} 2022, Pittsburgh, USA,
               May 25-27, 2022},
  year={2022}
}
```

## Requirements

* Python 3.8

## Tests

In the `counterfactuals` folder, run `python tests/regression_test.py` and `python tests/classifier_test.py`

## License

CC-BY-NC 4.0 (Attr Non-Commercial Inter.) (e.g., FAIR) licensed, as found in the LICENSE file.
