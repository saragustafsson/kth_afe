# -*- coding: utf-8 -*-

'''
Oscar Cleve, Sara Gustafsson, 2019

The code writes windows of raw data from the dataset to a csv-file.
'''

from storage import writeCsv, loadCsv, writeResults
from load_data import loadData
import pandas as pd


def main():
    directory = 'common'
    windowSize = 256
    windowSlide = 32
    removePart = 0.15
    dataInfo = 'windows_all'

    parameters = str(windowSize) + '_' + str(windowSlide)
    
    activities, data, totalDataPoints, labels = loadData("data\ADL_Dataset", windowSize, windowSlide, removePart)
    
    print("\nActivities:\n", activities)
    print("\nNumber of activities read: ", len(activities))
    print("Number of data points read: ", totalDataPoints)
    print("Number of readings read: ", len(data))
    print("Number of window labels read: ", len(labels), "\n")
    
    df_data = pd.DataFrame(data, columns=['x', 'y', 'z', 'id'])
    print("\ndf_data shape:")
    print(df_data.shape)
    
    writeCsv(directory, parameters, dataInfo, df_data)
    

if __name__ == '__main__':
    main()
