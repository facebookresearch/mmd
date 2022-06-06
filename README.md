# MMD: Machine Learning Model Diagnosis and Counterfactual Explanatins for Models of Code

Machine learning models often mispredict, and it is hard to tell when and why. 
This repository hosts two model diagnosis tools that support understanding the inner working of black-box models.

## MMD

We developed a technique, *MMD* ([rule_inductio](/rule_induction)), that systematically discovers rules that characterize a subset of the input space of a machine learning model where the model is more likely to mispredict.

This work has been published at the International Conference on Foundations in Software Engineering (FSE'21): J. Cito, I. Dillig, S. Kim, V. Murali, S. Chandra, [Explaining Mispredictions of Machine Learning Models using Rule Induction](https://github.com/facebookresearch/mmd/blob/main/paper/FSE21-ML-Misprediction-Preprint.pdf).

```bibtex
@inproceedings{explaining_mispredictions:21,
  title={Explaining mispredictions of machine learning models using rule induction},
  author={Cito, J{\"u}rgen and Dillig, Isil and Kim, Seohyun and Murali, Vijayaraghavan and Chandra, Satish},
  booktitle={Proceedings of the 29th ACM Joint Meeting on European Software Engineering Conference and Symposium on the Foundations of Software Engineering},
  pages={716--727},
  year={2021}
}
```

## Counterfactual Explanations for Models of Code

Counterfactual explanations ([counterfactuals](/counterfactuals)) constitute minimal changes to the input space under which the model
“changes its mind". The contrast between original input and perturbed input is considered an explanation.

This work has been published at the International Conference on Software Engineering (ICSE'22), Software Engineering in Practice: J. Cito, I. Dillig, V. Murali, S. Chandra, [Counterfactual Explanations for Models of Code](https://arxiv.org/pdf/2111.05711.pdf).

```bibtex
@inproceedings{code_counterfactuals:22,
  title={Counterfactual Explanations for Models of Code},
  author={Cito, J{\"u}rgen and Dillig, Isil and Murali, Vijayaraghavan and Chandra, Satish},
booktitle = {44th {IEEE/ACM} International Conference on Software Engineering:
               Software Engineering in Practice, {ICSE} {(SEIP)} 2022, Madrid, Spain,
               May 25-27, 2022},
  year={2022}
}
```



## Requirements

* Python 3.8
* Pandas

## License

Both projects are CC-BY-NC 4.0 (Attr Non-Commercial Inter.) (e.g., FAIR) licensed, as found in the LICENSE file.
