import torch

import config
from neural_network import NeuralNetwork
from readfile import readfile
from split_data import split_data
from trigrams import create_ngram, split_data_features_labels
from vectorization import *
#from traintest import *
from new_traintest import *
import time
from verbose import *
start = time.time()

def generate_trigram_model(eng_train, w2v_vectors, one_hot_encoded_vectors_eng):
    # create trigrams
    english_sentence_word_trigrams = create_ngram(eng_train)
    # get trigram vectors for all sentences
    english_sentence_vector_trigrams = make_vector_trigrams(english_sentence_word_trigrams, w2v_vectors, one_hot_encoded_vectors_eng)
    # create input features and labels out of eng_data for training the network
    X_list, Y_list = split_data_features_labels(english_sentence_vector_trigrams)

    #if config.process_unit == "gpu":
        # Using GPU (fast)
        #X = torch.cuda.FloatTensor(X_list) # gpu variable must have input type FloatTensor
        #Y = torch.cuda.FloatTensor(Y_list) 
        #trigram_eng_model = NeuralNetwork("gpu", lr=0.01)
    #else: 
        # Using CPU (slow)
        #X = torch.Tensor(X_list) 
        #Y = torch.Tensor(Y_list) 
        #trigram_eng_model = NeuralNetwork("cpu", lr=0.01)
    
    trigram_eng_model = NeuralNetwork(lr=0.01, device=config.process_unit)

    # The training for trigram model is done here
    #trigram_eng_model.train(X, Y, 600, len(Y[0]), 20)
    tstart = time.time()
    trigram_eng_model.train(X_list, Y_list, 600, len(Y_list))
    tstop = time.time()
    h, m, s = convert_time(tstart, tstop)
    print("Total run time for trigram model: {}:{}:{} ".format(h, m, s))
    return trigram_eng_model


def generate_translation_model(eng_train, french_train, w2v_vectors, one_hot_encoded_vectors_french):
    X_list, Y_list = make_translation_vectors(eng_train, french_train, w2v_vectors, one_hot_encoded_vectors_french)

    #if config.process_unit == "gpu":
        # Using GPU (fast)
        #X = torch.cuda.FloatTensor(X_list) # gpu variable must have input type FloatTensor
        #Y = torch.cuda.FloatTensor(Y_list) 
        #translation_model = NeuralNetwork("gpu", lr=0.01)
        
    #else: 
        # Using CPU (slow)
        #X = torch.Tensor(X_list) 
        #Y = torch.Tensor(Y_list) 
        #translation_model = NeuralNetwork("cpu", lr=0.01)

    translation_model = NeuralNetwork(lr=0.01, device=config.process_unit)

    # The training for translation model is done here
    #translation_model.train(X, Y, 600, len(Y[0]), 20)
    tstart = time.time()
    translation_model.train(X_list, Y_list, 600, len(Y_list))
    tstop = time.time()
    h, m, s = convert_time(tstart, tstop)
    print("Total run time for translation model: {}:{}:{} ".format(h, m, s))

    return translation_model


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
    #generate_one_hot_encoded_vectors
    eng_train, eng_test, french_train, french_test = split_data(english_data, french_data, test_size)

    # get vocabulary from training and testing data
    print("Generating vocabulary for target text...")
    eng_vocabulary = get_vocabulary(english_data)
    
    print("Generating vocabulary for source text...")
    french_vocabulary = get_vocabulary(french_data)

    # get one hot encoded vectors for training and testing data
    one_hot_encoded_vectors_eng = generate_one_hot_encoded_vectors(eng_vocabulary)
    one_hot_encoded_vectors_french = generate_one_hot_encoded_vectors(french_vocabulary)
    
    # get word2vec vectors for training and testing data
    w2v_vectors = get_w2v_vectors(w2v_model, eng_vocabulary)
    
    print("Training trigram model...")
    trigram_model = generate_trigram_model(eng_train, w2v_vectors, one_hot_encoded_vectors_eng)
    
    print("Training translation model...")
    translation_model = generate_translation_model(eng_train, french_train, w2v_vectors, one_hot_encoded_vectors_french)
    
    
    # import joblib
    # joblib.dump(X, "../temp_X.pkl")
    # joblib.dump(Y, "../temp_Y.pkl")
    # X = joblib.load("../temp_X.pkl")
    # Y = joblib.load("../temp_Y.pkl")

    nr_of_words = 50
    # training(translation_model, trigram_model, eng_train, french_train, w2v_vectors, one_hot_encoded_vectors_french, nr_of_words, eng_vocabulary)
    test_translation(eng_test, french_test, eng_vocabulary, french_vocabulary, w2v_vectors, one_hot_encoded_vectors_eng, one_hot_encoded_vectors_french, trigram_model, translation_model, config.process_unit)

# Time log and CUDA details
end = time.time()
h, m, s = convert_time(start, end)
print("---------------\nTotal run time: {}:{}:{} ".format(h, m, s))
#verbose_cuda()
