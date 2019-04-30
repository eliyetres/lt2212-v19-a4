import os
import sys
import argparse
import random
import time
import pickle
# from verbose import convert_time
# from readfile import readfile
# from split_data import split_data
from utils import convert_time, readfile, split_data, create_ngram, split_data_features_labels
from new_traintest import test_translation
from vectorization import load_gensim_model, remove_words, get_vocabulary, generate_indices, gen_tri_vec_split, get_w2v_vectors, generate_translation_vectors


parser = argparse.ArgumentParser(description="Feed forward neural networks.")

parser.add_argument("targetfile", type=str, default="UN-english", nargs='?', help="File used as target language.")
parser.add_argument("sourcefile", type=str, default="UN-french", nargs='?', help="File used as source language..")
parser.add_argument("googlemodelfile", type=str, default="GoogleNews-vectors-negative300.bin", nargs='?', help="Pre-trained word2vec 300-dimensional vectors.")
parser.add_argument("languagemodel", type=str, default="trained_language_model", nargs='?', help="Trained language model")
parser.add_argument("translationmodel", type=str, default="trained_translation_model", nargs='?', help="Trained translation model")
parser.add_argument("-T", "--top", metavar="T", dest="top_predicted", type=int, default=50, help="Top n predicted words (default 50).")
parser.add_argument("-P", "--processor", metavar="P", dest="processor", type=str, default="cpu", help="Select processing unit (default cpu).")

args = parser.parse_args()

# Variables
test_size = 0.2 # Test data %, default 20
p = args.processor

start = time.time()

print("Using {}.".format(args.processor))

print("Loading target language from {} and source language from {}.".format(args.targetfile, args.sourcefile))
target, source = readfile(args.targetfile, args.sourcefile)

print("Loading model from {}.".format(args.googlemodelfile))
pre_trained_model = load_gensim_model(args.googlemodelfile)

print("Removing words not found in the model.")
target_data, source_data = remove_words(target, source, pre_trained_model)

print("Splitting data into training and testing sets, {}/{}.".format((test_size*100), (100-(test_size*100))))
target_train, target_test, source_train, source_test = split_data(target_data, source_data, test_size)

print("Generating vocabulary for source text.")
source_vocab = get_vocabulary(source_data)

print("Generating vocabulary for target text.")
target_vocab = get_vocabulary(target_data)

print("Generating word indices.")
source_indices = generate_indices(source_vocab)
target_indices = generate_indices(target_vocab)

print("Fetching vectors from model.")
vectors = get_w2v_vectors(pre_trained_model, target_vocab)
 
print("Loading language model.")
with open(args.languagemodel, 'rb') as lmf:
    trigram_target_model = pickle.load(lmf)[1]

print("Loading translation model.")
with open(args.translationmodel, 'rb') as tmf:        
    translation_model = pickle.load(tmf)[1]

test_translation(target_test, source_test, target_vocab, source_vocab, vectors, target_indices, source_indices, trigram_target_model, translation_model, p, args.top_predicted)

stop = time.time()

hour, minute, second = (convert_time(start, stop))
print("Predicted {} sentences on {} hours, {} minutes and {} seconds".format(len(source_test), hour, minute, second))
