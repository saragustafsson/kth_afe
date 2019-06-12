# -*- coding: utf-8 -*-

'''
Oscar Cleve, Sara Gustafsson, 2019
 
This module loads raw data from files and returns labels and data split into windows.
The data should be stored in subdirectories under the input directory indicating the class of the data.
For example subdirectories of "walk", "drink-water" and "comb-hair" can be used.
When reading the data the labels are automaticly decided based on these directory names.

This program is written for data having three channels of raw data (x, y, z).
Important for the rest of the feature extraction to work is that a column 'id' is added to specify different windows.

Input: directory in which data is found, window size, window slide, part to remove at beginning/end of each recording (0-1)
'''

import codecs  
import os

'''
Input:  directory name (String),
        windowSize (Integer),
        window slide (Integer),
        remove part (Float) which is the part that is excluded at the beginning and end of each file
Output: activities: the names of the activities in data set (list of strings)
        data: raw data segmented in windows
        totalDataPoints: integer
        labels: list of class labels for each window
'''
def loadData (dirName, windowSize, slide, removePart):
    
    windowIndex = 0
    activities = []
    data = []
    totalDataPoints = 0
    labels = []

    for root, subfolders, files in os.walk(dirName):
        for subfolder in subfolders:
            subfolderPath = (os.path.join(root,subfolder))
            activity = subfolder
            activityFiles = []

            for file in os.listdir(subfolderPath):
                filePath = os.path.join(subfolderPath,file)
                activityFiles.append(filePath)

            activities.append(activity)
            activityIndex = activities.index(activity)
            activityDataPoints, activityData, windowIndex, labels = loadLabelData(activity, activityIndex, activityFiles, windowSize, slide, removePart, windowIndex, labels)
    
            totalDataPoints += activityDataPoints
            data += activityData
            
    return activities, data, totalDataPoints, labels


'''
Help function to loadData()
Loads all data files for a specific activity
''' 
def loadLabelData(activity, activityIndex, activityFiles, windowSize, slide, removePart, windowIndex, labels):

    g = 9.83 # gravitation
    activityDataPoints = 0
    activityData = []

    for file in activityFiles:
        fileData = []

        with codecs.open(file, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                activityDataPoints += 1
                realValues = []

                convertedValues = line.split()

                for value in convertedValues:
                    realValues.append(-1.5*g + (int(value)/63)*3*g)
                    
                realValues.append(0) # placeholder for window id
                    
                fileData.append(realValues)
            
        start = int(removePart*len(fileData))
        end = int((1-removePart)*len(fileData))
        usedFileData = fileData[start:end]

        splitFileData, windowIndex, labels = slidingWindow(usedFileData, activityIndex, windowSize, slide, windowIndex, labels) # returns array of arrays
        
        activityData += (splitFileData) # just add the windows to the data array
            
    return activityDataPoints, activityData, windowIndex, labels

        
'''
Help function to loadLabelData()
Split the input array into windows with given size and slide.
For each window add the activityIndex to the labels list.
Increase the windowIndex with the number of windows created.
'''
def slidingWindow(array, activityIndex, windowSize, slide, windowIndex, labels):
    
    numberOfWindows = int(((len(array) - windowSize) / slide) + 1) # the number of windows that will fit into the given array
    
    # this windowList will contain each window's data on each spot in the list
    # three dimensional matrix: number of windows x data points per window x data per datapoint
    windowList = []
    for i in range(numberOfWindows):
        start = 0 + i * slide
        stop = windowSize + i * slide
        window = array[start:stop]
        windowList.append(window)
        
    # we want to flatten the windowList so that each window's data follows the other ones'
    # to keep track of different windows we add the windowIndex as a column
    # two dimensional matrix: (number of windows * number of datapoints per window) x data per datapoint
    flatList = [] 
    for window in windowList:
        labels.append(activityIndex) # for each window we want to save the correct label in our labels array
        for line in window:
            line[3] = windowIndex
            lineCopy = line.copy()
            flatList.append(lineCopy)
            
        windowIndex += 1

    return flatList, windowIndex, labels