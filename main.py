import torch

import config
from neural_network import NeuralNetwork
from readfile import readfile
from split_data import split_data
from trigrams import create_ngram, split_data_features_labels
from vectorization import *
#from traintest import *
from new_traintest import *

# def generate_trigram_model(eng_train, w2v_vectors, one_hot_encoded_vectors_eng):
#     # create trigrams
#     english_sentence_word_trigrams = create_ngram(eng_train)
#     # get trigram vectors for all sentences
#     english_sentence_vector_trigrams = make_vector_trigrams(english_sentence_word_trigrams, w2v_vectors, one_hot_encoded_vectors_eng)
#     # create input features and labels out of eng_data for training the network
#     X_list, Y_list = split_data_features_labels(english_sentence_vector_trigrams)

#     if config.process_unit == "gpu":
#         # Using GPU (fast)
#         #X = torch.cuda.FloatTensor(X_list) # gpu variable must have input type FloatTensor
#         #Y = torch.cuda.FloatTensor(Y_list) 
#         trigram_eng_model = NeuralNetwork("gpu", lr=0.01)
#     else: 
#         # Using CPU (slow)
#         #X = torch.Tensor(X_list) 
#         #Y = torch.Tensor(Y_list) 
#         trigram_eng_model = NeuralNetwork("cpu", lr=0.01)
    
#     # The training for trigram model is done here
#     #trigram_eng_model.train(X, Y, 600, len(Y[0]), 20)
#     trigram_eng_model.train(X_list, Y_list, 600, len(Y_list[0]), 20)
#     return trigram_eng_model


# def generate_translation_model(eng_train, french_train, w2v_vectors, one_hot_encoded_vectors_french):
#     X_list, Y_list = make_translation_vectors(eng_train, french_train, w2v_vectors, one_hot_encoded_vectors_french)

#     if config.process_unit == "gpu":
#         # Using GPU (fast)
#         #X = torch.cuda.FloatTensor(X_list) # gpu variable must have input type FloatTensor
#         #Y = torch.cuda.FloatTensor(Y_list) 
#         translation_model = NeuralNetwork("gpu", lr=0.01)
        
#     else: 
#         # Using CPU (slow)
#         #X = torch.Tensor(X_list) 
#         #Y = torch.Tensor(Y_list) 
#         translation_model = NeuralNetwork("cpu", lr=0.01)

#     # The training for translation model is done here
#     #translation_model.train(X, Y, 600, len(Y[0]), 20)
#     translation_model.train(X_list, Y_list, 600, len(Y_list[0]), 20)

#     return translation_model


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
    
    print("Generating vocabulary for source text...")
    french_vocabulary = get_vocabulary(french_data)

    # get one hot encoded vectors for training and testing data
    #one_hot_encoded_vectors_eng = generate_one_hot_encoded_vectors(eng_vocabulary)
    #one_hot_encoded_vectors_french = generate_one_hot_encoded_vectors(french_vocabulary)

    #instead of one-hot vectors, save words as indices
    print("Generating word indices...")
    eng_indices, eng_len = generate_indices(eng_vocabulary)
    fr_indices, fr_len = generate_indices(french_vocabulary)
    
    # get word2vec vectors for training and testing data
    w2v_vectors = get_w2v_vectors(w2v_model, eng_vocabulary)
    
    # create trigrams
    print("Generating trigrams for English training data...")
    english_sentence_word_trigrams = create_ngram(eng_train)

    b = config.batch_size
    
    # ######## TRIGRAM MODEL ########
    # X_Y = generate_trigram_vector(english_sentence_word_trigrams, w2v_vectors, eng_indices, eng_len)
    # X, Y = split_data_features_labels(X_Y)
    
    print("Training trigram model...")
    trigram_eng_model = NeuralNetwork(config.process_unit, lr=0.01)
    # Initiate network weights
    
    init_sample = random.sample(english_sentence_word_trigrams, b)    
    X, Y = gen_tri_vec_split(init_sample, w2v_vectors, eng_indices, eng_len)   
    # Train translation model
    trigram_eng_model.start(X, Y, 600, len(Y[0]))
    for i in range(0, len(english_sentence_word_trigrams), b):  
        X, Y = gen_tri_vec_split(english_sentence_word_trigrams[i:i+b], w2v_vectors, eng_indices, eng_len)        
        trigram_eng_model.train(X, Y, 100)

    # ######## TRANSLATION MODEL ########     
    #trigram_model = generate_trigram_model(eng_train, w2v_vectors, one_hot_encoded_vectors_eng)    
    #print("Training translation model...")
    #translation_model = generate_translation_model(eng_train, french_train, w2v_vectors, one_hot_encoded_vectors_french)   

    print("Training translation model...")
    translation_model = NeuralNetwork(config.process_unit, lr=0.01)
    # Initiate network weights
    l = len(eng_train)    
    i_slice = random.randint(0,(l-b))    
    X_list, Y_list = generate_translation_vectors(eng_train[i_slice:i_slice+l], french_train[i_slice:i_slice+l], w2v_vectors, fr_indices, l)
    translation_model.start(X_list, Y_list, 600, len(Y_list[0]))  
    # Train translation model
    for i in range(0, len(eng_train), b):  
        X_list, Y_list = generate_translation_vectors(eng_train[i:i+b], french_train[i:i+b], w2v_vectors, fr_indices, l)
        translation_model.train(X_list, Y_list, 80)  
    
    
    # import joblib
    # joblib.dump(X, "../temp_X.pkl")
    # joblib.dump(Y, "../temp_Y.pkl")
    # X = joblib.load("../temp_X.pkl")
    # Y = joblib.load("../temp_Y.pkl")

    #nr_of_words = 50
    # training(translation_model, trigram_model, eng_train, french_train, w2v_vectors, one_hot_encoded_vectors_french, nr_of_words, eng_vocabulary)
    
    #test_translation(eng_test, french_test, eng_vocabulary, french_vocabulary, w2v_vectors, one_hot_encoded_vectors_eng, one_hot_encoded_vectors_french, trigram_model, translation_model, config.process_unit)

    test_new(eng_test, french_test, eng_vocabulary, french_vocabulary, w2v_vectors, eng_indices, fr_indices, eng_len, fr_len, trigram_eng_model, translation_model, config.process_unit, 50)
