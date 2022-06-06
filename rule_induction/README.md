# MMD: Machine Learning Misprediction Diagnoser

Machine learning models often mispredict, and it is hard to tell when and why. We developed a technique, *MMD*, that systematically discovers rules that characterize a subset of the input space of a machine learning model where the model is more likely to mispredict.

Our work has been published at the International Conference on Foundations in Software Engineering (FSE'21): J. Cito, I. Dillig, S. Kim, V. Murali, S. Chandra, [Explaining Mispredictions of Machine Learning Models using Rule Induction](https://github.com/facebookresearch/mmd/blob/main/paper/FSE21-ML-Misprediction-Preprint.pdf).

```bibtex
@inproceedings{explaining_mispredictions:21,
  title={Explaining mispredictions of machine learning models using rule induction},
  author={Cito, J{\"u}rgen and Dillig, Isil and Kim, Seohyun and Murali, Vijayaraghavan and Chandra, Satish},
  booktitle={Proceedings of the 29th ACM Joint Meeting on European Software Engineering Conference and Symposium on the Foundations of Software Engineering},
  pages={716--727},
  year={2021}
}
```

## Requirements

* Python 3.8
* Pandas

## License

MMD is CC-BY-NC 4.0 (Attr Non-Commercial Inter.) (e.g., FAIR) licensed, as found in the LICENSE file.
