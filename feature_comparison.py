# -*- coding: utf-8 -*-

'''
Oscar Cleve, 2019

This file makes a comparison of the features in two csv-documents.
It is a simple comparison of strings and a calculation of the numbers
of overlapping features.
'''

from storage import loadCsv, writeCsv, writeResults
from correlation import correlationFilter

import pandas as pd
import numpy as np


def main():

    freshFile = '256_32_bala_1.6e-05_feat_names_190509_1054.csv'
    rfFile = '256_32_-inf_1484_bala_feat_names_190519_1211.csv'

    freshFeatures = loadCsv('fresh', freshFile)
    rfFeatures = loadCsv('rf', rfFile)
    
    freshList = freshFeatures.values.tolist()
    rfList = rfFeatures.values.tolist()
    
    #Flattening of list
    freshFlat = []
    fftCounterFresh = 0
    for freshSublist in freshList:
        for val in freshSublist:
            freshFlat.append(val)
            if 'fft_coefficient__coeff' in val:
                fftCounterFresh = fftCounterFresh + 1
            
    rfFlat = []
    fftCounterRf = 0
    for rfSublist in rfList:
        for value in rfSublist:
            rfFlat.append(value)
            if 'fft_coefficient__coeff' in value:
                fftCounterRf = fftCounterRf + 1
    
    featureOverlap = list(set(freshFlat) & set(rfFlat))
    print('\nOverlaping features:\n' , featureOverlap)
    
    overlapPercentage = len(featureOverlap)/len(rfFlat)
    print("Number of features overlapping:", len(featureOverlap))
    
    print('\nPercentage of features overlapping:\n', overlapPercentage*100, '%\n')
    print('Percentage of features selected by FRESH being FFT:\n', (fftCounterFresh/len(freshFlat))*100 , '%')
    print('Percentage of features selected by RF being FFT:\n', (fftCounterRf/len(rfFlat))*100 , '%')


if __name__ == '__main__':
    main()
