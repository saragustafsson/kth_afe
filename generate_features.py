# -*- coding: utf-8 -*-

'''
Oscar Cleve, Sara Gustafsson, 2019

This code extracts features with TSFRESH on segmented raw data and saves them to file.
The code also contains functionality to make a classification instantly with all features extracted.
To get better understanding of data formats for input to functions see file load_data.py and the function loadData.
'''

'''
To profile memory usage: https://pypi.org/project/memory-profiler/
Install in conda prompt: conda install -c spyder-ide spyder-memory-profiler
If profiling should be done: uncomment import below and uncomment @profile before functions that should be profiled
'''
# from memory_profiler import profile

from storage import writeCsv, loadCsv, writeResults
from load_data import loadData
import pandas as pd

from tsfresh.feature_extraction import ComprehensiveFCParameters
from tsfresh.utilities.dataframe_functions import impute
from tsfresh import extract_features, extract_relevant_features, select_features

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


def main():
    directory = 'common'
    windowSize = 256
    windowSlide = 32
    removePart = 0.15
    
    testing = False # if True use a smaller set of data and don't save features to csv files
    classify = True # make classification and store results using ALL features if True
    #classWeight = None # balanced or None (for decision tree classifier)
    classWeight = 'balanced'

    parameters = str(windowSize) + '_' + str(windowSlide)
    
    activities, data, totalDataPoints, labels = loadData("data\ADL_Dataset", windowSize, windowSlide, removePart)
    
    print("\nActivities:\n", activities)
    print("\nNumber of activities read: ", len(activities))
    print("Number of data points read: ", totalDataPoints)
    print("Number of readings read: ", len(data))
    print("Number of window labels read: ", len(labels), "\n")


    if testing:
        N = 30
    else:
        N = len(labels)
    
    # convert data into a Pandas dataframe with the specified column names
    # the id column contains one number for each different window and is used to group datapoints for each window
    df_data = pd.DataFrame(data, columns=['x', 'y', 'z', 'id'])
    print("\ndf_data shape:")
    print(df_data.shape)
    
    # we extract the smaller set of data or if not specified the entire data set is used
    df_test = df_data.iloc[:windowSize*(N),:]
    print("\ndf_test shape:")
    print(df_test.shape)
    print("\ndf_test first 5 rows:")
    print(df_test.head())
    
    # convert label data to Pandas series
    y = pd.Series(labels[:N])
    print("\nShape of Y: ")
    print(y.shape)
    print(y.head())
    
    # Extract features on the windows
    X = extractFeatures(df_test)
    #X = loadCsv('common', '288_32_features_all_190409_1511.csv')
    
    # save all features to csv file
    if not testing:
        writeCsv(directory, parameters, 'features_all', X)
        writeCsv(directory, parameters, 'labels_all', y)

    ##### train classifier on all features #####

    if classify: 
        classification(parameters, classWeight, X, y, directory)
        
 
'''
This function extract features on the segmented data
If a different set of feature calcultares should be used this is changed here
Input:  raw window data (Dataframe) where:
        rows = number of datapoints, columns = data channels (e.g. x, y, z) and id column
        The id specifies what datapoints belong to the same window
Output: feature values (Dataframe) where:
        rows = number of windows, columns = number of features
'''
# @profile
def extractFeatures(rawData):
    print("\nSetting extraction settings")
    extraction_settings = ComprehensiveFCParameters()
    print("Before extracting features")
    X = extract_features(rawData, column_id='id', column_value = None, column_kind = None, impute_function=impute, default_fc_parameters=extraction_settings)
    print("After extracting features")
    print("Number of extracted features: {}.".format(X.shape[1]))
    print("\nShape of X: ")
    print(X.shape)
    
    return X


'''
This function makes a classification of data using a decision tree.
Input:  parameters thath should be included in file name of saved result (String),
        classWeight setting (None or 'balanced'),
        X = the data to be classified (Dataframe)
        y = correct labels (Series)
        directory in which to store results file (String)
Output: -
'''
# @profile
def classification(parameters, classWeight, X, y, directory):
    parameters += '_' + str(classWeight)[:4]
    print("\nSplitting data into train/test sets")
    # Random split in 20% testing data and 80% training data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2)
    
    print("Training classifier - ALL FEATURES")
    cl = DecisionTreeClassifier(class_weight=classWeight)
    cl.fit(X_train, y_train)
    resultsAll = classification_report(y_test, cl.predict(X_test))
    print(resultsAll)
    writeResults(directory, parameters, 'class', resultsAll)
    
    
if __name__ == '__main__':
    main()
