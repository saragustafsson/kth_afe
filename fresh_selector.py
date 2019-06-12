# -*- coding: utf-8 -*-

'''
Oscar Cleve, Sara Gustafsson, 2019

This file contatins the code for the FRESH selection process as well as the optional correlation filter.
There are options for loading prefiltered features straight into the correlation filter process.
'''

from storage import loadCsv, writeCsv, writeResults, saveModel
from correlation import correlationFilter

import pandas as pd
import numpy as np

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

from tsfresh.feature_extraction import ComprehensiveFCParameters
#from tsfresh.feature_selection import FeatureSignificanceTestsSettings
from tsfresh.utilities.dataframe_functions import impute
from tsfresh import extract_features, extract_relevant_features, select_features


def main():

    filterFeatures = True # specify True/False if you want to use TSFRESH feature filter or load pre-filtered features and only run the colleration filter
    removeCorrelated = True # specify True/False if you want to remove correlated features or just run the TSFRESH filter process

    directory = 'fresh'
    windowSize = 256
    windowSlide = 32
    classWeight = 'balanced' # balanced or None
    fdrLevel = 0.0004 #fdr level for the FRESH selector
    parameters = str(windowSize) + '_' + str(windowSlide) + '_' + str(classWeight)[:4] + '_' + str(fdrLevel)
    threshold = 0.15 # treshold for the correlation filter
    
    if filterFeatures:

        featureFile = '256_32_features_all_190408_1123.csv'
        labelFile = '256_32_labels_all_190408_1123.csv'

        X = loadCsv('common', featureFile) # returns dataframe
        y_temp = loadCsv('common', labelFile) # returns dataframe
        y = y_temp.iloc[:,0]

        print('\nShape of X: ')
        print(X.shape)

        y = pd.Series(y)
        print('Shape of Y (as series): ')
        print(y.shape)

        print('\nSplitting data into train/test sets')
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2)

        ##### 1. multi class feature selection #####

        print('\nShape of X_test: ', X_test.shape)
        print('Shape of y_test: ', y_test.shape)
        print()

        relevant_features = set()

        # The multi class problem is converted into a binary problem when filtering features
        # Reference: https://github.com/blue-yonder/tsfresh/blob/master/notebooks/human_activity_recognition_multi_class_example.ipynb
        for label in y.unique():
            y_train_binary = y_train == label
            X_train_filtered = select_features(X_train, y_train_binary, ml_task='classification', fdr_level = fdrLevel)
            print('Number of relevant features for class {}: {}/{}'.format(label, X_train_filtered.shape[1], X_train.shape[1]))
            relevant_features = relevant_features.union(set(X_train_filtered.columns))

        print('Number of relevant features (union of all classes): ', len(relevant_features))
        X_train_filtered = X_train[list(relevant_features)]
        X_test_filtered = X_test[list(relevant_features)]

        featureSeries = pd.Series(list(relevant_features))

        # writeCsv(directory, parameters, 'feat_train', X_train_filtered)
        # writeCsv(directory, parameters, 'feat_test', X_test_filtered)
        # writeCsv(directory, parameters, 'labels_train', y_train)
        # writeCsv(directory, parameters, 'labels_test', y_test)
        writeCsv(directory, parameters, 'feat_names', featureSeries)

        ##### train classifier on filtered features #####

        print('\nTraining classifier - FILTERED FEATURES')
        cl = DecisionTreeClassifier(class_weight=classWeight)
        cl.fit(X_train_filtered, y_train)
        prediction = cl.predict(X_test_filtered)
        
        result = classification_report(y_test, prediction)
        print(result)
        accScore = accuracy_score(y_test, prediction)
        print('Accuracy:', accScore)
        
        #writeResults(directory, parameters, 'class', result)

    else: # load pre-filtered features

        X_train_filtered = loadCsv(directory, '128_32_bala_feat_train_190407_1802.csv')
        X_test_filtered = loadCsv(directory, '128_32_bala_feat_test_190407_1803.csv')
        y_train = loadCsv(directory, '128_32_bala_labels_train_190407_1803.csv').iloc[:,0]
        y_test = loadCsv(directory, '128_32_bala_labels_test_190407_1803.csv').iloc[:,0]

        print('\nShape of loaded:')
        print('X_train_filtered: ', X_train_filtered.shape)
        print('X_test_filtered: ', X_test_filtered.shape)
        print('y_train: ', y_train.shape)
        print('y_test: ', y_test.shape)


    if removeCorrelated:

        ##### 2. remove correlated features #####

        parameters = str(windowSize) + '_' + str(windowSlide) + '_' + str(classWeight)[:4] + '_corr_0' + str(threshold*10)

        keptFeatures = correlationFilter(X_train_filtered, threshold)

        relevant_features = set(keptFeatures.columns)

        X_train_uncorrelated = X_train_filtered[list(relevant_features)]
        X_test_uncorrelated = X_test_filtered[list(relevant_features)]

        print('\nShape of uncorrelated X_train: ', X_train_uncorrelated.shape)
        print('Shape of uncorrelated X_test: ', X_test_uncorrelated.shape)

        featureSeries = pd.Series(list(relevant_features))

        # writeCsv(directory, parameters, "feat_train", X_train_uncorrelated)
        # writeCsv(directory, parameters, "feat_test", X_test_uncorrelated)
        # writeCsv(directory, parameters, "labels_train", y_train)
        # writeCsv(directory, parameters, "labels_test", y_test)
        # writeCsv(directory, parameters, 'feat_names', featureSeries)

        ##### train classifier on uncorrelated filtered features #####

        print('\nTraining classifier - UNCORRELATED FILTERED FEATURES, threshold = ', threshold)
        cl = DecisionTreeClassifier(class_weight = classWeight)
        cl.fit(X_train_uncorrelated, y_train)
        prediction = cl.predict(X_test_uncorrelated)

        #saveModel(cl, parameters)

        resultUncorr = classification_report(y_test, prediction)
        print(resultUncorr)
        
        result = classification_report(y_test, prediction)
        print(result)
        accScore = accuracy_score(y_test, prediction)
        print('Accuracy:', accScore)
        
        # writeResults(directory, parameters, 'class', resultUncorr)


if __name__ == '__main__':
    main()
