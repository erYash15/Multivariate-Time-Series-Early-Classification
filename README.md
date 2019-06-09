# Multivariate-Time-series-classification

## Project name: Early Classification of Time Series Data

### Description: 

Project is divided into 3 Sub-Objectives:

_Objective 1_ - Discover the internal Characteristics of MTS (Multivariate Time-Series) data and enhance the interpretability of classification. **Extact Feature Candidate** of each variable.

_Objective 2_ - **Mine Core Features** from the extracted features using **Greedy Method** and **SI Clustering**. Core feature is any shapelet extremely useful in classification.

_Objective 3_ - **Early Classification** using **Rule Based Method** and **QBC Method**.

### Table of Contents:

Files associated with Objective 1:

- [feature_extration.py](https://github.com/erYash15/Multivariate-Time-series-early-classification/blob/master/feature_extration.py)

Files associated with Objective 2:

- [feature_selection_greedy_1.py](https://github.com/erYash15/Multivariate-Time-series-early-classification/blob/master/feature_selection_greedy_1.py)
- [feature_selection_greedy_2.py](https://github.com/erYash15/Multivariate-Time-series-early-classification/blob/master/feature_selection_greedy_2.py)

Files associated with Objective 3:

- [early_classification_MCFEC_QBC.py](https://github.com/erYash15/Multivariate-Time-series-early-classification/blob/master/early_classification_MCFEC_QBC.py)

### Pre-requisites and Installation:
This project requires **Python** and the following Python libraries installed:

- [NumPy](http://www.numpy.org/)
- [Pandas](http://pandas.pydata.org/)
- [matplotlib](http://matplotlib.org/)
- [scikit-learn](http://scikit-learn.org/stable/)

### Usage:

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
python anyone_file_from_objective_3
```

This is final early classififcation with earliness and accuracy. 



### Contributing:



### Credits:

Project is based on the paper "Early classification on multivariate time series". Author Guoliang He, Yong Duan, Rong Peng, Xiaoyuan Jing, Tieyun Qian, Lingling Wang.

- [Early classification on multivariate time series](https://dl.acm.org/citation.cfm?id=2841855)

### License:



