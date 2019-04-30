import re
from sklearn.model_selection import train_test_split
from nltk import ngrams
import numpy as np


def convert_time(start, stop): 
    total_seconds = stop-start
    seconds = total_seconds % (24 * 3600) 
    hour = seconds // 3600
    seconds %= 3600
    minutes = seconds // 60
    seconds %= 60      
    
    return round(hour), round(minutes), round(seconds)\


def readfile(eng_file, fre_file):
    english_words = []
    french_words = []
    with open(eng_file, encoding="utf8") as engtext:
        with open(fre_file, encoding="utf8") as fretext:
            for eng_line, fre_line in zip(engtext, fretext):
                eng_line = re.sub(
                    r'([\"\<\>\(\)\:\[\]\%\/\=\?\!\;\n\*\.\,]|[0-9])', '', eng_line).lower()
                fre_line = re.sub(
                    r'([\"\<\>\(\)\:\[\]\%\/\=\?\!\;\n\*\.\,]|[0-9])', '', fre_line).lower()
                # Split string into list of words
                eng_words = eng_line.split(" ")
                fre_words = fre_line.split(" ")
                eng_words = [x for x in eng_words if x]  # Remove empty strings
                fre_words = [x for x in fre_words if x]
                e = len(eng_words)
                f = len(fre_words)
                shortest_sentence = min(e, f)
                # Cut the longer sentence
                eng_words = eng_words[:shortest_sentence]
                fre_words = fre_words[:shortest_sentence]
                english_words.append(eng_words)
                french_words.append(fre_words)
                
    return english_words, french_words


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


def create_ngram(vocabulary):
    """ Takes a list of word form the vocabulary.
    Returns trigrams as tuples in a list of lists (one for every sentence) padded with starting tag. """
    all_trigrams = []
    for sentence in vocabulary:
        trigram_sentence = []
        # Padding n-grams with start tag
        generated_grams = ngrams(
            sentence, 3, pad_left=True, left_pad_symbol='<start>', pad_right=False)
        for each_gram in generated_grams:
            trigram_sentence.append(each_gram)
        all_trigrams.append(trigram_sentence)
    return all_trigrams


def split_data_features_labels(trigram_sents):
    X = []
    Y = []
    
    for sent in trigram_sents:
        for trigram in sent:
            X.append(np.hstack((trigram[0], trigram[1])))
            Y.append(trigram[-1])

    return X, Y