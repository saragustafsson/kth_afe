# -*- coding: utf-8 -*-

'''
Oscar Cleve, Sara Gustafsson, 2019

This code loads extracted features and makes a selection with the Random Forest-method.
'''

from storage import *
import pandas as pd
import numpy as np
from joblib import dump
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier


def main():

    directory = 'rf'
    windowSize = 256
    windowOverlap = 32
    threshold = -np.inf # -np.inf if maxFeatures is used
    maxFeatures = 10
    classWeight = 'balanced' # 'balanced' or None
    parameters = str(windowSize) + '_' + str(windowOverlap) + '_' + str(threshold) + '_' + str(maxFeatures) + '_' + str(classWeight)[:4]

    featureFile = '256_32_features_all_190408_1123.csv'
    labelFile = '256_32_labels_all_190408_1123.csv'

    X = loadCsv('common', featureFile)
    y = loadCsv('common', labelFile)
    y = y.iloc[:,0]

    print('\nShape of X: ')
    print(X.shape)

    print('\nShape of Y: ')
    print(y.shape)

    print('\nSplitting data into train/test sets')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2)

    #Creating the rfEstimator object for SelectFromModel
    rf = RandomForestClassifier(n_estimators = 100, max_depth = 40, max_features = 0.2, min_samples_split = 2, criterion = 'entropy')
    sfm = SelectFromModel(rf, prefit=False, threshold=threshold, max_features=maxFeatures)
    sfm.fit_transform(X_train, y_train)

    featureIndices = sfm.get_support(indices=True)
    featureNames = list(X_train.columns.values[featureIndices])
    featureSeries = pd.Series(featureNames)
    print('\nSelected features:\n')
    print(featureSeries)

    X_train_filtered = X_train.iloc[:,featureIndices]
    X_test_filtered = X_test.iloc[:,featureIndices]

    print('\nNumber of relevant features: ', len(featureIndices))

    #writeCsv(directory, parameters, 'feat_train', X_train_filtered)
    #writeCsv(directory, parameters, 'feat_test', X_test_filtered)
    #writeCsv(directory, parameters, 'labels_train', y_train)
    #writeCsv(directory, parameters, 'labels_test', y_test)
    #writeCsv(directory, parameters, 'feat_names', featureSeries)

    ##### train classifier on filtered features #####

    print('\nTraining classifier - FILTERED FEATURES')
    cl = DecisionTreeClassifier(class_weight=classWeight)
    cl.fit(X_train_filtered, y_train)

    #saveModel(cl, parameters)
    
    prediction = cl.predict(X_test_filtered)
    
    f1Result = classification_report(y_test, prediction)
    print(f1Result)
    
    accResult = accuracy_score(y_test, prediction)
    print('Accuracy in percent:', 100*accResult)
    #writeResults(directory, parameters, 'class', f1Result)

    print('Number of samples classified: ', len(y_test))


if __name__ == '__main__':
    main()
