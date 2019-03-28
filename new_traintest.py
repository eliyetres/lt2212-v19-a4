#Training and testing script
from neural_network import *
import torch
import operator
import numpy as np
from trigrams import create_ngram


def get_predicted_word(predictions, vocab):
    
    #This code is correct, but the problem is that the size of predictions
    #is larger than the size of vocab, therefore it keeps looping out of index

    max_index, max_value = max(enumerate(predictions), key=operator.itemgetter(1))

    print("Max index, length of vocabulary, {}".format(max_index, len(vocab)))
    #print("Max value: {}".format(max_value))
    return vocab[max_index]


def get_score_for_word(pred, first_french_word_index):
    return pred[first_french_word_index]


def get_top_n_predictions(next_word_pred, n=50):

    #sorted_preds = sorted(next_word_pred, reverse=False)

    # next_word_pred is a Tensor
    # use sorting method of torch, returning indices
    sorted_preds, indices = torch.sort(next_word_pred, descending=True)

    top_n_indices = []
    
    for i in range(0, n):
        #index = next_word_pred.index( sorted_preds[i] )

        index = indices[0][i]
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

    #Output variables
    #Measurements should be accuracy, precision, recall and F1_score
    total_translations = 0
    correct_translations = 0
    
    # create trigrams out of eng test data# create trigrams out of eng test data
    eng_trigrams = create_ngram(eng_test)
    
    # create a list of w2v vectors of all english words in vocab
    english_vectors = []
    for word in eng_vocab:
        vect = w2v_vectors[word]
        english_vectors.append(vect)
    english_vectors_tensor = torch.Tensor(english_vectors)

    for index in range(len(french_test)):

        print("Sentence {}".format(index))
        print("French sentence: {}".format(french_test[index]))
        print("English sentence: {}".format(eng_test[index]))

        print("Length of sentence: {}".format(len(french_test[index])))
        
        translated_english_words = []

        
        #get first english original word
        first_english_original = eng_test[index][0]
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
        print("First word translated into: {}".format(translated_word))

        translated_english_words.append(translated_word)
        
        # calculates scores
        total_translations += 1
        if translated_word == first_english_original:
            correct_translation +=1

        # the trigram now will be <start> and translated_word
        first_word = '<start>'
        second_word = translated_word
        #for word_index in french_test[index]:
        for word_index in range(1, len(french_test[index])): #Looping from 1 to end of sentence, skipping index 0

            #print("Word {}, {}".format(word_index+1, french_test[index][word_index+1]))
            print("Word {}, {}".format(word_index, french_test[index][word_index]))

            
            #bigram = np.concatenate((w2v_vectors[first_word], w2v_vectors[second_word]))
            bigram = np.hstack((w2v_vectors[first_word], w2v_vectors[second_word]))
            bigram = torch.Tensor([bigram])

            try:
                next_word_pred = trigram_model.predict(bigram)
            except:
                next_word_pred = trigram_model.predict([bigram])
            
            top_50_prediction_indices = get_top_n_predictions(next_word_pred, n=50)

            #next_french_word = french_test[index][word_index+1]
            next_french_word = french_test[index][word_index]            

            #next_french_word_index = french_vocab[next_french_word]
            next_french_word_index = french_vocab.index(next_french_word)
            
            max_pred_score = -1
            translated_vector = None

            #Make a list
            eng_word_vectors = []

            for pred_index in top_50_prediction_indices:
                eng_word = eng_vocab[pred_index]
                eng_word_vector = w2v_vectors[eng_word]

                #Append vector to the list
                eng_word_vectors.append(eng_word_vector)

                #Make it a tensor out of the list
                vector_tensor = torch.Tensor(eng_word_vectors)

                try:
                    translated_pred = translation_model.predict(vector_tensor) #was eng_word_vector
                except:
                    translated_pred = translation_model.predict([vector_tensor])

                # ******* Something goes wrong here, the result from the model is larger then the size of the vocab *****
                translated_pred = translated_pred[0]
                    
                score = get_score_for_word(translated_pred, next_french_word_index)
                if score > max_pred_score:
                    score = max_pred_score
                    translated_vector = translated_pred

            translated_word = get_predicted_word(translated_vector, eng_vocab)
            print("Translated into: {}".format(translated_word))
            translated_english_words.append(translated_word)
            
            # calculates scores
            total_translations += 1
            if translated_word == next_english_original:
                correct_translation +=1

            first_word = second_word
            second_word = translated_word

        print("Translated sentence: {}".format(' '.join(translated_english_words)))
