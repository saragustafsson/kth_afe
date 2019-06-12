# -*- coding: utf-8 -*-

'''
Oscar Cleve, Sara gustafsson, 2019

This code searches through different hyperperameters for the Random Forest selector with Randomized Search.
'''

from storage import loadCsv, writeCsv, writeResults
import pandas as pd
import numpy as np
#from joblib import dump

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import classification_report
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier


def main():

    directory = 'rf'
    windowSize = 256
    windowOverlap = 32
    threshold = None # -np.inf if maxFeatures is used
    maxFeatures = 10
    classWeight = None # balanced or None
    parameters = str(windowSize) + '_' + str(windowOverlap) + '_' + str(threshold) + '_' + str(maxFeatures) + '_' + str(classWeight)[:4]

    f1 = '256_32_features_all_190408_1123.csv'
    f2 = '256_32_labels_all_190408_1123.csv'

    X = loadCsv('common', f1)
    y = loadCsv('common', f2)
    y = y.iloc[:,0]

    print('\nShape of X: ')
    print(X.shape)

#    y = pd.Series(labels)
    print('\nShape of Y: ')
    print(y.shape)

    #Creating the rfEstimator object for SelectFromModel
    rf = RandomForestClassifier(n_estimators=100)
    print(rf.get_params())
    print()

    # use a full grid over all parameters
    param_grid = {'max_depth': [None, 10, 20, 30, 40],
                  'max_features': ['auto', 'log2', None, 0.2, 10],
                  'min_samples_split': [2, 10, 18, 26, 34],
                  'criterion': ['gini', 'entropy']}

    # run grid search
    grid_search = RandomizedSearchCV(rf, n_iter=40, param_distributions=param_grid, cv=5, scoring = 'f1_macro', n_jobs = -1)
    grid_search.fit(X, y)

    report(grid_search.cv_results_)

# Utility function to report best scores
def report(results, n_top=5):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print('Model with rank: {0}'.format(i))
            print('Mean validation score: {0:.3f} (std: {1:.3f})'.format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print('Parameters: {0}'.format(results['params'][candidate]))
            print('')

if __name__ == '__main__':
    main()
