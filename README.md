# KTH_AFE_bachelor_thesis
Clean repository for Oscar Cleve and Sara Gustafsson's bachelor thesis project at KTH.

## Data
The raw data is found in the data directory.

## Feature extraction
Generate all features by executing the generate_features.py file. Make sure the execution saves your extracted features, that saving is not not commented in the code, since this can take quite some time.

## Feature selection
This repository contains two separate methods for feature selection after the initial pool of features has been generated.

1) The FRESH algorithm implemented in TSFRESH combined with a correlation filter. This is in the fresh_selector.py file.

2) Random forest selector which can be found and run using the RandomForest_Selector.py file.

Both of these files also contains classification using a decision tree classifier and will present results after execution using the selected features.

## Helping functions/modules
The following files are only used imported in other programs:
correlation.py
load_data.py
storage.py

## Other
classification_only.py is used to extract only selected features and make a classifcation using a pre-trained classifier.

correlation_group.py can be used to evaluate correlation in a dataset, using the correlation.py functions.

feature_compariosn.py is used to compare two lists of selected features.

find_hyper_parameters.py is used to search for best hyperparameters to use in the Random Forest feature selector method.
