#Training and testing script
from neural_network import *
import torch
import operator
import numpy as np
from trigrams import create_ngram

def training(t_model, l_model, eng_sent, french_sent, w2v_vectors, onehot_french, num_of_words, vocabulary): 

    #print(w2v_vectors) = dict with word : array
    #print(onehot_french) = dict with word : array
    
    #Variables for input
    hidden_layer_size = 600
    epochs = 4000
    #num_of_classes = len(Y[0])

    #Output variables
    accuracy = 0
    precision = 0
    recall = 0
    F1_score = 0

    english_vectors = []

    #Sentences

    # create trigrams                                                                                                                                                               
    english_sentence_word_trigrams = create_ngram(eng_sent)
    nr_of_sent = len(english_sentence_word_trigrams)

    #Get w2v for vocabulary
    for word in vocabulary:
        vect = w2v_vectors[word]
        english_vectors.append(vect)

    english_vectors_tensor = torch.Tensor(english_vectors)

        
    #For each sentence
    for i in range(nr_of_sent):
        
        english = eng_sent[i]
        french = french_sent[i]
        english_sentence_trigram = english_sentence_word_trigrams[i] #should loop over trigrams?

        print(english)
        print(french)
        
        predicted_french = {}
        chosen_vector = []

        #Get the onehot vector for the third word in the sentence:
        firstfrench = french[2]
        firstfrench_ohv = onehot_french[firstfrench]

        print(firstfrench)
        #print(firstfrench_ohv)
        
        #Run translations model with the english_vectors
        predicted_output = t_model.predict(english_vectors_tensor)
        print(predicted_output)
        print(torch.sum(predicted_output))
        
        #Choose the vector that predicts the French word we have
        #chosen_vector.append(predicted_french[firstfrench_ohv])

        #print("The English w2v vector {} is corresponding to the French onehot {} for the word {}".format(chosen_vector, actual_french_vector, firstfrench))



        # **** Steps ahead ***
        #Run the language model with the chosen vector and the start tag
        
        #Find the word of the argmax of the factor of these two

        #For the rest of the words in the sentence:
        #for i in range(len(sentence)[3:]):
            #Use the start symbol and the word found above
            #Predict the num_of_words next English words

            #Use those words above to maximize the translation model probability
            #for the next French word



    #Calculate and print output measurements here
    #Also print out the translations


def get_predicted_word(predictions, vocab):
    max_index, max_value = max(enumerate(predictions), key=operator.itemgetter(1))
    print(max_value)
    return vocab[max_index]


def get_score_for_word(pred, first_french_word_index):
    return pred[first_french_word_index]


def get_top_n_predictions(next_word_pred, n=50):
    sorted_preds = sorted(next_word_pred, reverse=False)
    top_n_indices = []
    for i in range(0, n):
        index = next_word_pred.index( sorted_preds[i] )
        top_n_indices.append(index)
    return top_n_indices


# for testing
# create trigrams out of eng test data
# for sentence in test data:
#       get first french word
#       create a list of w2v vectors of all english words in vocab
#       feed this to translation model, get output
#       choose the english word which gives this french word as output -> how to do this?
#       Take <start> and this english word, feed to trigram model
def test_translation(eng_test, french_test, eng_vocab, french_vocab, w2v_vectors, one_hot_eng, one_hot_french, trigram_model, translation_model):
    # create trigrams out of eng test data# create trigrams out of eng test data
    eng_trigrams = create_ngram(eng_test)
    
    # create a list of w2v vectors of all english words in vocab
    english_vectors = []
    for word in eng_vocab:
        vect = w2v_vectors[word]
        english_vectors.append(vect)
    english_vectors_tensor = torch.Tensor(english_vectors)

    for index in range(len(french_test)):
        # get first french word
        first_french_word = french_test[index][0]
        first_french_word_index = french_vocab.index(first_french_word)
        # get predicted french words for all english words
        predictions = translation_model.predict(english_vectors_tensor)
        translated_vector = None
        max_pred_score = -1
        # cycle through all predictions
        for pred in predictions:
            # get score for this word's prediction from one hot output
            pred_score = get_score_for_word(pred, first_french_word_index)
            if pred_score > max_pred_score:
                max_pred_score = pred_score
                translated_vector = pred
            # get corresponding french word of predicted vector
            # predicted_word = get_predicted_word(pred, french_vocab)
            # if predicted word = first french word, we need this english word; break here
            # if predicted_word == first_french_word:
            #     translated_vector = pred
            #     break

        # get the corresponding english word of this vector
        translated_word = get_predicted_word(translated_vector, eng_vocab)
        print(translated_word)

        # the trigram now will be <start> and translated_word
        first_word = '<start>'
        second_word = translated_word
        for word_index in french_test[index]:
            bigram = np.concatenate((w2v_vectors[first_word], w2v_vectors[second_word]))
            try:
                next_word_pred = trigram_model.predict(bigram)
            except:
                next_word_pred = trigram_model.predict([bigram])

            top_50_prediction_indices = get_top_n_predictions(next_word_pred, n=50)
            next_french_word = french_test[index][word_index+1]
            next_french_word_index = french_vocab[next_french_word]
            max_pred_score = -1
            translated_vector = None
            for pred_index in top_50_prediction_indices:
                eng_word = eng_vocab[pred_index]
                eng_word_vector = w2v_vectors[eng_word]
                try:
                    translated_pred = translation_model.predict(eng_word_vector)
                except:
                    translated_pred = translation_model.predict([eng_word_vector])

                score = get_score_for_word(translated_pred, next_french_word_index)
                if score > max_pred_score:
                    score = max_pred_score
                    translated_vector = translated_pred

            translated_word = get_predicted_word(translated_vector, eng_vocab)
            print(translated_word)
            first_word = second_word
            second_word = translated_word