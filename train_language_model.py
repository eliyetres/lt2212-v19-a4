import os
import sys
import argparse
import random
import time
import pickle
from verbose import convert_time
from neural_network import NeuralNetwork
from readfile import readfile
from split_data import split_data
from trigrams import create_ngram, split_data_features_labels
from vectorization import load_gensim_model, remove_words, get_vocabulary, generate_indices, gen_tri_vec_split, get_w2v_vectors, generate_translation_vectors
from neural_network import NeuralNetwork

parser = argparse.ArgumentParser(description="Feed forward neural networks.")

parser.add_argument("targetfile", type=str, default="UN-english-sample-small.txt", nargs='?', help="File used as target language.")
parser.add_argument("sourcefile", type=str, default="UN-french-sample-small.txt", nargs='?', help="File used as source language..")
parser.add_argument("modelfile", type=str, default="GoogleNews-vectors-negative300.bin", nargs='?', help="Pre-trained word2vec 300-dimensional vectors.")
parser.add_argument("trainedmodelfile", type=str, default="trained_language_model.pickle", nargs='?', help="Trained language model")

#parser.add_argument("-M", "--model", metavar="M", dest="model", type=str, default="", help="Trained language model (default empty string)")
parser.add_argument("-B", "--batch", metavar="B", dest="batch", type=int,default=100, help="Batch size used for for training the neural network (default 100).")
parser.add_argument("-E", "--epoch", metavar="E", dest="epoch", type=int,default=20, help="Number or epochs used for training the neural network (default 20).")
parser.add_argument("-L", "--layer", metavar="L", dest="layer", type=int, default=600, help="Layer size for the neural network (default 600).")
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

start = time.time()

print("Using {}.".format(args.processor))

print("Loading target language from {} and source language from {}.".format(args.targetfile, args.sourcefile))
target, source = readfile(args.targetfile, args.sourcefile)

print("Loading model from {}.".format(args.modelfile))
pre_trained_model = load_gensim_model(args.modelfile)

print("Removing words not found in the model.")
target_data, source_data = remove_words(target, source, pre_trained_model)

print("Splitting data into training and testing sets, {}/{}.".format(round(100-(test_size*100)), round(test_size*100)))
target_train, target_test, source_train, source_test = split_data(target_data, source_data, test_size)

if b >= len(target_train):
    exit("Error: training batch must be lower than training data.")

print("Generating vocabulary for target text.")
target_vocab = get_vocabulary(target_data)

print("Generating trigrams for target training data.")
target_trigrams = create_ngram(target_train)

print("Generating word indices.")
target_indices = generate_indices(target_vocab)

print("Fetching vectors from model.")
vectors = get_w2v_vectors(pre_trained_model, target_vocab)

print("Feeding training data in batches of size: {}".format(b))

print("Training language model.")

#If no model is incoming, create one:
if os.path.exists(args.trainedmodelfile) and os.path.getsize(args.trainedmodelfile) <= 0:

    print("Initiating an empty model")
    trigram_target_model = NeuralNetwork(p, r)
    start_batch = 0
    text = "original"
        
else:
    with open(args.trainedmodelfile, 'rb') as mf:
        print("Loading model from file {}.".format(mf))
        #trigram_target_model = NeuralNetwork(p, r)
        content = pickle.load(mf)
        start_batch = content[0]
        trigram_target_model = content[1]
        
print(start_batch)
    
init_sample = random.sample(target_trigrams, b)

# Initiate network weights from sample 
X, Y = gen_tri_vec_split(init_sample, vectors, target_indices)   

# Train translation model
trigram_target_model.start(X, Y, layer_size, len(Y[0]))

#For each batch...
for i in range(start_batch, len(target_trigrams), b):
    print("Trigrams {} - {}".format(i, i+b))
    X, Y = gen_tri_vec_split(target_trigrams[i:i+b], vectors, target_indices)        
    trigram_target_model.train(X, Y, epochs)

    #...write model to file
    print("Writing model to {}, having processed {} trigrams.".format(args.trainedmodelfile, i+b))
    with open(args.trainedmodelfile, 'wb') as tmf:
        pickle.dump([i+b, trigram_target_model], tmf)
        #pickle.dump([i+b, "test"], tmf)

stop = time.time()

hour, minute, second = (convert_time(start, stop))
print("Trained {} sentences on {} hours, {} minutes and {} seconds".format(len(source_data), hour, minute, second))
