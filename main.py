from readfile import readfile
from split_data import split_data 
from vectorization import *
from trigrams import create_ngram, split_data_features_labels

import config


if __name__ == '__main__':
    
    print("Loading gensim model...")
    w2v_model = load_gensim_model(config.google_model)
    
    # getting data from readfile.py
    print("Reading data from source and target files...")
    english, french = readfile(config.english_filename, config.french_filename)
    test_size = 0.2

    # check data against the model and remove words not found
    print("Removing words not found in the model")
    english_data, french_data = remove_words(english, french, w2v_model)
    
    # split the data into training and testing sets
    print("Splitting data into training and testing sets...")
    eng_train, eng_test, french_train, french_test = split_data(english_data, french_data, test_size)

    # get vocabulary from training and testing data
    print("Generating vocabulary for target text...")
    eng_vocabulary = get_vocabulary(english_data)
    print(eng_vocabulary)
    
    print("Generating vocabulary for source text...")
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
    
    # create trigram
    english_sentence_word_trigrams = create_ngram(eng_train)
    
    # get trigram vectors for all sentences
    english_sentence_vector_trigrams = make_vector_trigrams(english_sentence_word_trigrams, w2v_vectors)

    # create input features and labels out of eng_data for training the network
    X, Y = split_data_features_labels(english_sentence_vector_trigrams)

    