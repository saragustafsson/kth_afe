# -*- coding: utf-8 -*-

'''
Oscar Cleve, Sara Gustafsson, 2019

This code contains function to apply correlation filter on data.
It can also present visual analysis of the correlation if this file is executed.

Input: dataframe with all data, shape: number of features x number of samples (windows)
Output: dataframe were the columns having a correlation > threshold have been removed
'''

import matplotlib.pyplot as plt
import seaborn as sns
from storage import *

def correlationFilter(data, threshold):
    
    print("Generating correlation matrix...")
    correlationMatrix = data.corr(method='pearson') # pandas.DataFrame.corr(), method: pearson (default), kendall, spearman
    #correlationMatrix = loadCsv('common', '_correlationMatrix_190509_1120.csv')
    print('Correlation calculated.')
    #writeCsv('common', '', 'correlationMatrix', correlationMatrix)

    generateHistogram(correlationMatrix)
    generateHeatmap(correlationMatrix)
    
    
    # find the correlated features
    # reference: https://stackabuse.com/applying-filter-methods-in-python-for-feature-selection/
    correlatedFeatures = set()
    print("Searching through all columns for correlated features...")
    for i in range(len(correlationMatrix.columns)):
        for j in range(i):
            if abs(correlationMatrix.iloc[i,j]) > threshold:
                correlatedFeatures.add(correlationMatrix.columns[i])

    # remove the correlated features
    print("Drop correlated features: ", len(correlatedFeatures))
    data.drop(labels = correlatedFeatures, axis=1, inplace = True)

    return data


# Compute and show histogram  (https://matplotlib.org/api/_as_gen/matplotlib.pyplot.hist.html)
def generateHistogram(correlationMatrix):

    print('Computing histogram...')
    correlationList = convertToList(correlationMatrix)
    plt.hist(correlationList, bins = 160)
    plt.rcParams["figure.figsize"] = [12, 6] # set plot figure size
    plt.xlabel('Correlation')
    plt.ylabel('Occurences')
    plt.title('Correlation distribution')
    plt.grid(True)
    plt.show()
    

# Help function to generateHistrogram()  
def convertToList(frame):
    
    valuesList = []
    
    for i in range(frame.shape[0]):
        for j in range(i+1, frame.shape[1]):
            valuesList.append(frame.iloc[i,j])
            
    absList = [abs(number) for number in valuesList]

    average = sum(absList) / len(absList)
    print('Average of correlationlist: ', average)
    print('Length of list: ', len(absList))
    
    return valuesList
    
    
# Show a heatmap of correlation matrix    
def generateHeatmap(correlationMatrix):
    
    ax = sns.heatmap(correlationMatrix, annot = True, cbar = False, linecolor = 'white', linewidth=0.5)
    plt.show()


# to test the correlation function
def main():
    
    directory = 'fresh'
    threshold = 0.1
    X_train_filtered = loadCsv(directory, '128_32_bala_feat_train_190404_1535.csv')
    keptFeatures = correlationFilter(X_train_filtered, threshold)

 
if __name__ == '__main__':
    main()