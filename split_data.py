#split_data.py

from readfile import readfile

import numpy as np
from sklearn.model_selection import train_test_split


def split_data(target, source, size):
    
    #Printing the lengths of target and source data
    print("Total length of target language: {}".format(len(target)))
    print("Total length of source language: {}".format(len(source)))
    
    #Splitting data into sets
    #print("Splitting data into training and tests sets...")
    targetTrain, targetTest, sourceTrain, sourceTest = train_test_split(target, source, test_size=size)

    #Printing the lengths of the sets
    print("Length of target training set: {}".format(len(targetTrain)))
    print("Length of target test set: {}".format(len(targetTest)))
    print("Length of source training set: {}".format(len(sourceTrain)))
    print("Length of source test set: {}".format(len(sourceTest)))

    return targetTrain, targetTest, sourceTrain, sourceTest


#Specifying test size in percentage
# test_size = 0.2

#Running split_data
# split_data(target, source, test_size)
