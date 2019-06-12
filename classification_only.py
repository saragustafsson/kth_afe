# -*- coding: utf-8 -'*-

'''
Oscar Cleve, Sara Gustafsson, 2019

This program extracts only the features that have been selected by a previous feature selector.
These feature names are stored in a txt file.
A classification is done using a decision tree. The trained model is loaded - not trained in this program.
'''

from storage import *
from correlation import *
import pandas as pd
import numpy as np
#from joblib import load

from tsfresh.feature_extraction import ComprehensiveFCParameters
from tsfresh.utilities.dataframe_functions import impute
from tsfresh import extract_features, extract_relevant_features, select_features, feature_extraction

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier

def main():

    dataDirectory = 'rf'
    windowSize = 256
    windowSlide = 32
    classWeight = 'balanced' # 'balanced' or None
    threshold = -np.inf # RF: -np.inf if maxFeatures is used, FRESH: the correlation threshold (float)

    rf = True

    if rf:
        maxFeatures = 30
        parameters = str(windowSize) + '_' + str(windowSlide) + '_' + str(threshold) + '_' + str(maxFeatures) + '_' + str(classWeight)[:4]
    else: # fresh
        fdrLevel = 0.05
        parameters = str(windowSize) + '_' + str(windowSlide) + '_' + str(threshold) + '_' + str(fdrLevel) + '_' + str(classWeight)[:4]


    testWindowsDocName = '256_32_-inf_10_bala_feat_train_190506_2327.csv'
    testLabelDocName = '256_32_-inf_10_bala_labels_train_190506_2327.csv'
    classifierFile = '256_32_-inf_10_bala_DT_model_190506_2327.joblib'

    ### CHANGE HERE TO THE FEATURES THAT YOU WANT TO EVALUATE ###
    selectedFeaturesDirectory = 'rf'
    selectedFeatureDocName = '256_32_-inf_10_bala_feat_names_190506_2323.csv'


    ###############################################################################

    X_test_windows = loadCsv(dataDirectory, testWindowsDocName)
    y_test = loadCsv(dataDirectory, testLabelDocName)
    y_test = y_test.iloc[:,0]

    selectedFeatureNames = loadCsv(selectedFeaturesDirectory, selectedFeatureDocName)
    selectedFeatureNames = selectedFeatureNames.iloc[:,0]

    #removing z-data because we don't have features for that axis
    #X_test_windows = X_test_windows.iloc[:,[0,1,3]]

    kind_to_fc_parameters = feature_extraction.settings.from_columns(selectedFeatureNames)

    X_test = extract_features(X_test_windows, column_id='id', column_value = None, column_kind = None, impute_function=impute, kind_to_fc_parameters=kind_to_fc_parameters)

    correlationFilter(X_test, 1.2)

    print('\n\nNumber of samples: ' , len(y_test),'\n')

    cl = loadModel(classifierFile)
    prediction = cl.predict(X_test)
    result = classification_report(y_test, prediction)
    print(result)

    #writeResults(dataDirectory, parameters, 'class', result)

if __name__ == '__main__':
    main()
