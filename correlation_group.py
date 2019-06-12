# -*- coding: utf-8 -*-

'''
Sara Gustafsson, 2019

Compute the correlation between data in a group of selected features.
'''

from storage import *
from correlation import *
import pandas as pd
import numpy as np

from tsfresh.feature_extraction import ComprehensiveFCParameters
from tsfresh.utilities.dataframe_functions import impute
from tsfresh import extract_features, extract_relevant_features, select_features, feature_extraction


def main():

    dataDirectory = 'common'
    testWindowsDocName = '256_32_windows_all_190430_0950.csv'

    ### CHANGE HERE TO THE FEATURES THAT YOU WANT TO EVALUATE ###
    # selectedFeaturesDirectory = 'rf'
    # selectedFeatureDocName = '_256_32_-inf_10_bala_feat_names_190509_1126.csv'
    selectedFeaturesDirectory = 'fresh'
    selectedFeatureDocName = '_256_32_bala_corr_01.0_feat_names_190509_1048.csv'

    ###############################################################################

    X_test_windows = loadCsv(dataDirectory, testWindowsDocName)

    selectedFeatureNames = loadCsv(selectedFeaturesDirectory, selectedFeatureDocName)
    selectedFeatureNames = selectedFeatureNames.iloc[:,0]

    # removing data if we don't have features for that axis
    # Select axis from: 0, 1, 2, 3 (3 should always be included since it is the id)
    X_test_windows = X_test_windows.iloc[:,[0,3]]

    kind_to_fc_parameters = feature_extraction.settings.from_columns(selectedFeatureNames)

    X_test = extract_features(X_test_windows, column_id='id', column_value = None, column_kind = None, impute_function=impute, kind_to_fc_parameters=kind_to_fc_parameters)

    correlationFilter(X_test, 1.2)


if __name__ == '__main__':
    main()
