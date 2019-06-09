#Multivariate-Time-series-classification

##Project name: Early Classification of Time Series Data

###Description: 

Project is divided into 3 sub-objectives:
Objective 1: Discover the internal Characteristics of MTS (Multivariate Time-Series) data and enhance the interpretability of classification. EXTRACT FEATURE CANDIDATE of each variable.
OBJECTIVE 2: MINE CORE FEATURES from the extracted features using GREEDY METHOD and SI CLUSTERING. Core feature is any shapelet extremely useful in classification.
OBJECTIVE 3: EARLY CLASSIFICATION using RULE BASED METHOD and QBC METHOD.

###Table of Contents:

Files associated with Objective 1:

- [feature_extration.py]

Files associated with Objective 2:

- [feature_selection_greedy_1.py]
- [feature_selection_greedy_2.py]

Files associated with Objective 3:

- [early_classification_MCFEC_QBC.py]

###Pre-requisites and Installation:
This project requires **Python** and the following Python libraries installed:

- [NumPy](http://www.numpy.org/)
- [Pandas](http://pandas.pydata.org/)
- [matplotlib](http://matplotlib.org/)
- [scikit-learn](http://scikit-learn.org/stable/)

###Usage:

In a terminal or command window, navigate to the top-level project directory `Multivariate-Time-series-classification/` (that contains this README) and run command in sequence:

```bash
python <anyone_file_from_objective_1>
```

This takes time, then
```bash
python <files_from_objective_2>
```

either greedy or SI method files only one by one, then
```bash
python <anyone_file_from_objective_3>
```

This is final early classififcation with earliness and accuracy. 



###Contributing:


###Credits:

Project is based on the paper "Early classification on multivariate time series". Author Guoliang He, Yong Duan, Rong Peng, Xiaoyuan Jing, Tieyun Qian, Lingling Wang.

- [Early classification on multivariate time series](https://dl.acm.org/citation.cfm?id=2841855)

###License:



