import os
import sys
import argparse
import random
from neural_network import NeuralNetwork
from readfile import readfile
from split_data import split_data
from trigrams import create_ngram, split_data_features_labels
from new_traintest import test_new
from vectorization import load_gensim_model, remove_words, get_vocabulary, generate_indices, gen_tri_vec_split, get_w2v_vectors, generate_translation_vectors
from neural_network import NeuralNetwork

parser = argparse.ArgumentParser(description="Feed forward neural networks.")

parser.add_argument("targetfile", type=str, default="UN-english", help="File used as target language.")
parser.add_argument("sourcefile", type=str, default="UN-french", help="File used as source language..")
parser.add_argument("modelfile", type=str, default="GoogleNews-vectors-negative300.bin", help="Pre-trained word2vec 300-dimensional vectors.")
parser.add_argument("-B", "--batch", metavar="B", dest="batch", type=int,default=200, help="Batch size used for for training the neural network (default 100).")
parser.add_argument("-E", "--epoch", metavar="E", dest="epoch", type=int,default=20, help="Number or epochs used for training the neural network (default 200).")
parser.add_argument("-L", "--layer", metavar="L", dest="layer", type=int, default=600, help="Layer size for the neural network (default 600).")
parser.add_argument("-T", "--top", metavar="T", dest="top_predicted", type=int, default=50, help="Top n predicted words (default 50).")
parser.add_argument("-P", "--processor", metavar="P", dest="processor", type=str, default="cpu", help="Select processing unit (default cpu).")
parser.add_argument("-R", "--learning_rate", metavar="R", dest="learning_rate", type=float, default=0.01, help="Optimizer learning rate (default 0.01).")

args = parser.parse_args()

# Variables
test_size = 0.2 # Test data %, default 20
layer_size = args.layer
epochs = args.epoch
b = args.batch
p = args.processor
r = args.learning_rate

print("Using {}.".format(args.processor))

print("Loading target language from {} and source language from {}.".format(args.targetfile, args.sourcefile))
target, source = readfile(args.targetfile, args.sourcefile)

print("Loading model from {}.".format(args.modelfile))
pre_trained_model = load_gensim_model(args.modelfile)

print("Removing words not found in the model.")
target_data, source_data = remove_words(target, source, pre_trained_model)

print("Splitting data into training and testing sets, {}/{}.".format((test_size*100), (100-(test_size*100))))
target_train, target_test, source_train, source_test = split_data(target_data, source_data, test_size)

if b >= len(target_train):
    exit("Error: training batch must be lower than training data.")

print("Generating vocabulary for source text.")
source_vocab = get_vocabulary(source_data)

print("Generating vocabulary for target text.")
target_vocab = get_vocabulary(target_data)

print("Generating trigrams for target training data.")
target_trigrams = create_ngram(target_train)

print("Generating word indices.")
source_indices, source_len = generate_indices(source_vocab)
target_indices, target_len = generate_indices(target_vocab)

print("Fetching vectors from model.")
vectors = get_w2v_vectors(pre_trained_model, target_vocab)

print("Feeding training data in batches of size: {}".format(b))

print("Training language model.")
trigram_target_model = NeuralNetwork(p, r)
init_sample = random.sample(target_trigrams, b)
# Initiate network weights from sample 
X, Y = gen_tri_vec_split(init_sample, vectors, target_indices)   
# Train translation model
trigram_target_model.start(X, Y, layer_size, len(Y[0]))
for i in range(0, len(target_trigrams), b):
    X, Y = gen_tri_vec_split(target_trigrams[i:i+b], vectors, target_indices)        
    trigram_target_model.train(X, Y, epochs)

print("Training translation model.")
translation_model = NeuralNetwork(p, r)
# Initiate network weights
#l = len(target_indices)    
#i_slice = random.randint(0,(l-b))
start_i = random.randint(0,len(target_train)-b) # random start position
end_i = start_i+b
#X_list, Y_list = generate_translation_vectors(target_train[i_slice:i_slice+b], source_train[i_slice:i_slice+b], vectors, source_indices)
X_list, Y_list = generate_translation_vectors(target_train[start_i:end_i], source_train[start_i:end_i], vectors, source_indices)
translation_model.start(X_list, Y_list, layer_size, len(Y_list[0]))  
# Train translation model
for i in range(0, len(target_train), b):  
    X_list, Y_list = generate_translation_vectors(target_train[i:i+b], source_train[i:i+b], vectors, source_indices)
    translation_model.train(X_list, Y_list, epochs)  

test_new(target_test, source_test, target_vocab, source_vocab, vectors, target_indices, source_indices, trigram_target_model, translation_model, p, args.top_predicted)