import os
import re
import sys
import time
import pickle
import random
import argparse
from torch.cuda import device_count
from utils import convert_time, readfile, split_data, create_ngram, split_data_features_labels
from vectorization import load_gensim_model, remove_words, get_vocabulary, generate_indices, gen_tri_vec_split, get_w2v_vectors, generate_translation_vectors
from neural_network import NeuralNetwork


def train_language_model(start, trainedmodelfile, p, r, b, target_trigrams, target_indices, vectors, layer_size, epochs):
    print("Training language model.")

    # If no model is incoming, create one:
    if not os.path.isfile(trainedmodelfile) or os.path.getsize(trainedmodelfile) <=0:

        print("Initializing an empty model")
        trigram_target_model = NeuralNetwork(p, r)
        startpoint = 0

        init_sample = random.sample(target_trigrams, b)

        # Initiate network weights from sample
        X, Y = gen_tri_vec_split(init_sample, vectors, target_indices)

        # Train language model
        trigram_target_model.start(X, Y, layer_size, len(Y[0]))
        
    else:
        with open(trainedmodelfile, 'rb') as mf:
            print("Loading model from file {}.".format(mf))
            content = pickle.load(mf)
            startpoint = content[0]
            trigram_target_model = content[1]            
        
    # For each batch...
    for i in range(startpoint, len(target_trigrams), b):
        print("Trigrams {} - {} out of {}".format(i, i+b, len(target_trigrams)))
        X, Y = gen_tri_vec_split(target_trigrams[i:i+b], vectors, target_indices)        
        trigram_target_model.train(X, Y, epochs)

        # ...write model to file
        print("Writing model to {}, having processed {} trigrams.".format(trainedmodelfile, i+b))
        with open(trainedmodelfile, 'wb') as tmf:
            pickle.dump([i+b, trigram_target_model], tmf)

        t = time.time()
        h, m, s = (convert_time(start, t))
        print("Time passed: {}:{}:{}\n".format(h, m, s))

            
    stop = time.time()
    hour, minute, second = (convert_time(start, stop))
    print("Trained {} sentences on {} hours, {} minutes and {} seconds".format(len(target_trigrams), hour, minute, second))
    print("A trigram model is saved to the file {}".format(trainedmodelfile))


def train_translation_model(start, trainedmodelfile, p, r, b, source_indices, source_train, target_train, vectors, layer_size, epochs):
    print("Training translation model.")

    #If no model is incoming, create one:
    if not os.path.isfile(trainedmodelfile) or os.path.getsize(trainedmodelfile) <=0:

        print("File not found")
        print("Initiating an empty model")
        translation_model = NeuralNetwork(p, r)
        startpoint = 0

        start_i = random.randint(0,len(target_train)-b) # random start position
        end_i = start_i+b

        X_list, Y_list = generate_translation_vectors(target_train[start_i:end_i], source_train[start_i:end_i], vectors, source_indices)
        
        # Train translation model
        translation_model.start(X_list, Y_list, layer_size, len(Y_list[0]))

    else:
        with open(trainedmodelfile, 'rb') as mf:
            print("Loading model from file {}.".format(mf))    
            content = pickle.load(mf)
            startpoint = content[0]
            translation_model = content[1]

    #For each batch...
    for i in range(startpoint, len(target_train), b):
        print("Sentences {} - {} out of {}".format(i, i+b, len(target_train)))
        X_list, Y_list = generate_translation_vectors(target_train[i:i+b], source_train[i:i+b], vectors, source_indices)
        translation_model.train(X_list, Y_list, epochs)

        #...write model to file                                                                                                                                                     
        print("Writing model to {}, having processed {} sentences.".format(trainedmodelfile, i+b))
        with open(trainedmodelfile, 'wb') as tmf:
            pickle.dump([i+b, translation_model], tmf)    

        t = time.time()
        h, m, s = (convert_time(start, t))
        print("Time passed: {}:{}:{}\n".format(h, m, s))
            
    stop = time.time()
    hour, minute, second = (convert_time(start, stop))
    print("Ran {} sentences on {} hours, {} minutes and {} seconds".format(len(target_train), hour, minute, second))
    print("A translation model is saved to the file {}".format(trainedmodelfile))


parser = argparse.ArgumentParser(description="Feed forward neural networks.")

parser.add_argument("targetfile", type=str, default="UN-english.txt", nargs='?', help="File used as target language.")
parser.add_argument("sourcefile", type=str, default="UN-french.txt", nargs='?', help="File used as source language..")
parser.add_argument("modelfile", type=str, default="GoogleNews-vectors-negative300.bin", nargs='?', help="Pre-trained word2vec 300-dimensional vectors.")
parser.add_argument("trainedmodelfile", type=str, default="trained_model", nargs='?', help="File used as output for the trained model")
parser.add_argument("-M", "--modeltype", metavar="M", dest="model_type", type=int, default=0, help="Choose whether to train translation (1) or trigram (0) model")
parser.add_argument("-B", "--batch", metavar="B", dest="batch", type=int,default=100, help="Batch size used for for training the neural network (default 100).")
parser.add_argument("-E", "--epoch", metavar="E", dest="epoch", type=int,default=20, help="Number or epochs used for training the neural network (default 20).")
parser.add_argument("-L", "--layer", metavar="L", dest="layer", type=int, default=600, help="Layer size for the neural network (default 600).")
parser.add_argument("-P", "--processor", metavar="P", dest="processor", type=str, default="cpu", help="Select processing unit (default cpu).")
parser.add_argument("-R", "--learning_rate", metavar="R", dest="learning_rate", type=float, default=0.01, help="Optimizer learning rate (default 0.01).")
parser.add_argument("-S", "--test_size", metavar="T", dest="test_size", type=float, default=0.2, help="Size in percentage of test set (default 0.2).")

args = parser.parse_args()

# Variables
#test_size = 0.2 # Test data %, default 20
test_size = args.test_size
layer_size = args.layer
epochs = args.epoch
b = args.batch
p = args.processor
r = args.learning_rate
model_type = args.model_type

start = time.time()

if test_size < 0 or test_size > 1:
    exit("Error: Test size must be a number between 0 and lower than 1, e.g. 0.2")

if epochs < 0:
    exit("Error: Number of epochs can't be negative")

if r < 0 or r > 1:
    exit("Error: Learning rate must be a float between 0 and lower than 1, e.g. 0.01")

processor_valid = False
if p.lower() == "cpu":
    processor_valid = True
else:
    gpu_rg = r'cuda\:(\d{1,2})'
    m = re.search(gpu_rg, p, flags=re.I)
    if m:
        gpu_num = int(m.group(1))
        if gpu_num <= device_count() and gpu_num > 0:
            processor_valid = True

if processor_valid is False:
    exit("Processor type is invalid - only 'cuda' and 'cpu' are valid device types")

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


# train trigram model
if model_type == 0:
    print("Generating vocabulary for target text.")
    target_vocab = get_vocabulary(target_data)

    print("Generating trigrams for target training data.")
    target_trigrams = create_ngram(target_train)

    print("Generating word indices.")
    target_indices = generate_indices(target_vocab)

    print("Fetching vectors from model.")
    vectors = get_w2v_vectors(pre_trained_model, target_vocab)

    print("Feeding training data in batches of size: {}".format(b))
    train_language_model(start, args.trainedmodelfile, p, r, b, target_trigrams, target_indices, vectors, layer_size, epochs)

# train translation model
elif model_type == 1:
    print("Generating vocabulary for source text.")
    source_vocab = get_vocabulary(source_data)

    print("Generating vocabulary for target text.")
    target_vocab = get_vocabulary(target_data)

    print("Generating word indices.")
    source_indices = generate_indices(source_vocab)

    print("Fetching vectors from model.")
    vectors = get_w2v_vectors(pre_trained_model, target_vocab)

    print("Feeding training data in batches of size: {}".format(b))
    train_translation_model(start, args.trainedmodelfile, p, r, b, source_indices, source_train, target_train, vectors, layer_size, epochs)

# else
else:
    exit("Error: Model type can only be 0 (training trigrams) or 1 (training translation)")
