# Multivariate-Time-Series-Classification

## Project name: Early Classification of Time Series Data

### Description: 

Project is divided into 3 Sub-Objectives:

_Objective 1_ - Discover the internal Characteristics of MTS (Multivariate Time-Series) data and enhance the interpretability of classification. **Extract Feature Candidate** of each variable.

_Objective 2_ - **Mine Core Features** from the extracted features using **Greedy Method** and **SI Clustering**. Core feature is any shapelet extremely useful in classification.

_Objective 3_ - **Early Classification** using **Rule Based Method** and **QBC Method**.

### Table of Contents:

Files associated with Objective 1:

- [feature_extration.py](https://github.com/erYash15/Multivariate-Time-series-early-classification/blob/master/feature_extration.py)

Files associated with Objective 2:

- [feature_selection_greedy_1.py](https://github.com/erYash15/Multivariate-Time-series-early-classification/blob/master/feature_selection_greedy_1.py)
- [feature_selection_greedy_2.py](https://github.com/erYash15/Multivariate-Time-series-early-classification/blob/master/feature_selection_greedy_2.py)
- [SI_feature_selection](https://github.com/erYash15/Multivariate-Time-Series-Early-Classification/blob/master/SI_feature_selection.py)

Files associated with Objective 3:

- [early_classification_MCFEC_QBC.py](https://github.com/erYash15/Multivariate-Time-series-early-classification/blob/master/early_classification_MCFEC_QBC.py)
- [early_classification_MCFEC_Rule_Based.py](https://github.com/erYash15/Multivariate-Time-series-early-classification/blob/master/early_classification_MCFEC_Rule_Based.py)


### Pre-requisites and Installation:
This project requires **Python** and the following Python libraries installed:

- [NumPy](http://www.numpy.org/)
- [Pandas](http://pandas.pydata.org/)
- [matplotlib](http://matplotlib.org/)
- [scikit-learn](http://scikit-learn.org/stable/)

### Usage:

#### Use the main.py file (contains all the subcodes combined.)

In a terminal or command window, navigate to the top-level project directory `Multivariate-Time-series-classification/` (that contains this README) and run command in sequence:

```bash
python anyone_file_from_objective_1.py
```

_This may take time, then do_
```bash
python files_from_objective_2.py
```

_either greedy or SI method files only one by one, then do_
```bash
python anyone_file_from_objective_3.py
```

_This is final early classififcation with earliness and accuracy._

### Results

![Datasets](https://user-images.githubusercontent.com/34357926/102748576-6e4bb280-4388-11eb-8ff0-2376ef519a85.png)
![Summary Part 2](https://user-images.githubusercontent.com/34357926/102748577-6ee44900-4388-11eb-814f-fa8986ba208f.png)
![Summary Part 3](https://user-images.githubusercontent.com/34357926/102748573-6c81ef00-4388-11eb-900c-efb769a60829.png)

### Credits:

Project is based on the paper "[Early classification on multivariate time series](https://dl.acm.org/citation.cfm?id=2841855)". Author Guoliang He, Yong Duan, Rong Peng, Xiaoyuan Jing, Tieyun Qian, Lingling Wang.



### License:

To cite either a computer program or piece of source code you will need the following information:

Yash Gupta<br />Early Classification of Time Series Data<br />https://github.com/erYash15/Multivariate-Time-Series-Early-Classification
