# -*- coding: utf-8 -*-

'''
Oscar Cleve, Sara Gustafsson, 2019

This code only contains help functions for other programs.
The functions load and store data such as generated features or classifyer model from runnings in files.

This file will be working with the following directories which need to exist before execution:
    - runnings
    - runnings/dt_models
    - runnings/<your chosen directory name*>
    - runnings/<your chosen directory name*>/results

* use these to for example separate different methods runings and results
'''

import pandas as pd
import datetime
from joblib import dump, load

on_linux = False

# Input: the name of the directory in "runnings" where file is found (String), the name of the file (String)
# Output: file content (Dataframe)
def loadCsv(directory, fileName):

    global on_linux

    if on_linux:
        fileReading = pd.read_csv (r'runnings/' + directory + '/' + fileName)
    else:
        fileReading = pd.read_csv (r'runnings\\' + directory + '\\' + fileName)

    return fileReading


# Input: the name of the directory in "runnings" where file should be stored (String), parameters to include in file name (String), data info to include in the file name (String), data to be saved (Dataframe)
# Output: -
def writeCsv(directory, parameters, dataInfo, data):

    global on_linux

    timestamp = datetime.datetime.now().strftime("%y%m%d_%H%M")
    #fileName = dataInfo+'_wSize_'+str(windowSize)+'_wOverlap_'+str(windowOverlap)+'_'+timestamp+'.csv'
    fileName = parameters + '_' + dataInfo + '_' + timestamp + '.csv'
    if on_linux:
        data.to_csv (r'runnings/' + directory + '/' + fileName, index = None, header=True)
    else:
        data.to_csv (r'runnings\\' + directory + '\\' + fileName, index = None, header=True)


# NOTE: this function takes data as a string as input, not as a dataframe.
# Input: the name of the directory in "runnings" where file should be stored (String), parameters to include in file name (String), data info to include in the file name (String), data to be saved (String)
# Output: -
def writeResults(directory, parameters, dataInfo, data):

    global on_linux

    timestamp = datetime.datetime.now().strftime("%y%m%d_%H%M")
    #fileName = 'result_'+dataInfo+'_wSize_'+str(windowSize)+'_wOverlap_'+str(windowOverlap)+'_'+timestamp+'.txt'
    fileName = parameters + '_' + dataInfo + '_' + timestamp + '.txt'

    if on_linux:
        f = open('runnings/' + directory + '/results/' + fileName,"w+")
    else:
        f = open('runnings\\' + directory + '\\results\\' + fileName,"w+")

    f.write(data)
    f.close()

# Input: the trained classifyer to save, parameters to include in file name (String)
# Output: -
def saveModel(classifier, parameters):

    global on_linux

    timestamp = datetime.datetime.now().strftime("%y%m%d_%H%M")

    fileName = parameters + '_DT_model_' + timestamp + '.joblib'

    if on_linux:
        dump(classifier, 'runnings/dt_models/' + fileName)
    else:
        dump(classifier, 'runnings\dt_models\\' + fileName)


# Input: the file contains the saved model (String)
# Output: the classifyer
def loadModel(filename):

    global on_linux

    if on_linux:
        classifier = load('runnings/dt_models/' + filename)
    else:
        classifier = load('runnings\dt_models\\' + filename)

    return classifier
