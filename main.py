from readfile import readfile
from split_data import split_data 
from vectorization import *

import config


if __name__ == '__main__':
    
    print("Loading gensim model...")
    w2v_model = load_gensim_model(config.google_model)
    
    # getting data from readfile.py
    print("Reading data from source and target files...")
    english_data, french_data = readfile(config.english_filename, config.french_filename)
    test_size = 0.2

    # split the data into training and testing sets
    print("Splitting data into training and testing sets...")
    eng_train, eng_test, french_train, french_test = split_data(english_data, french_data, test_size)

    # get vocabulary from training and testing data
    print("Generating vocabulary for source text...")
    eng_vocabulary = get_vocabulary(english_data)
    print(eng_vocabulary)
    
    print("Generating vocabulary for target text...")
    french_vocabulary = get_vocabulary(french_data)

    # get one hot encoded vectors for training and testing data
    one_hot_encoded_vectors_eng = generate_one_hot_encoded_vectors(eng_vocabulary)
    one_hot_encoded_vectors_french = generate_one_hot_encoded_vectors(french_vocabulary)
    # for index in range(len(one_hot_encoded_vectors_eng)):
    #     print(eng_vocabulary[index])
    #     print(one_hot_encoded_vectors_eng[index])

    # get word2vec vectors for training and testing data
    w2v_vectors = get_w2v_vectors(w2v_model, eng_vocabulary)
    print("One hot encoded vector for 'the': {}".format(one_hot_encoded_vectors_eng['the']))
    print("Word2vec vector for 'the': {}".format(w2v_vectors['the']))