#split_data.py

from readfile import readfile

import numpy as np
from sklearn.model_selection import train_test_split


def split_data(size):

    #Getting data from readfile.py
    target, source = readfile("UN-english.txt", "UN-french.txt")

    #Printing the lengths of target and source data
    print("Total length of target language: ".format(len(t)))
    print("Total length of source language: ".format(len(s)))
    
    #Splitting data into sets
    print("Splitting data into training and tests sets...")
    targetTrain, targetTest, sourceTrain, sourceTest = train_test_split(target, source, test_size=size)

    #Printing the lengths of the sets
    print("Length of target training set: ".format(len(targetTrain)))
    print("Length of target test set: ".format(len(targetTest)))
    print("Length of source training set: ".format(len(sourceTrain)))
    print("Length of source test set: ".format(len(sourceTest)))

    return targetTrain, targetTest, sourceTrain, sourceTest


#Specifying test size in percentage
test_size = 0.2

#Running split_data
split_data(target, source, test_size)
