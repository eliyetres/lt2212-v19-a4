#Training and testing script
from neural_network import *
import torch
import operator
import numpy as np
from trigrams import create_ngram
from sklearn.metrics import classification_report


def make_tensor(vector_list, unit):

    if unit == "gpu":
        # Using GPU (fast)                                                                                                                                                          
        X = torch.cuda.FloatTensor(vector_list) # gpu variable must have input type FloatTensor                                                                                           
    else:
        # Using CPU (slow)                                                                                                                                                          
        X = torch.Tensor(vector_list)

    return X


def get_predicted_word(predictions, vocab):
    
    #This code is correct, but the problem is that the size of predictions
    #is larger than the size of vocab, therefore it keeps looping out of index

    max_index, max_value = max(enumerate(predictions), key=operator.itemgetter(1))

    print("Max index {}, length of vocabulary, {}".format(max_index, len(vocab)))
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
        # index = next_word_pred.index( sorted_preds[i] )

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

def test_translation(eng_test, french_test, eng_vocab, french_vocab, w2v_vectors, one_hot_eng, one_hot_french, trigram_model, translation_model, unit):

    # Output variables
    # Measurements should be accuracy, precision, recall and F1_score
    actual_translations = []
    predicted_translations = []
    
    # create trigrams out of eng test data# create trigrams out of eng test data
    eng_trigrams = create_ngram(eng_test)
    
    # create a list of w2v vectors of all english words in vocab
    english_vectors = []
    for word in eng_vocab:
        vect = w2v_vectors[word]
        english_vectors.append(vect)
    #english_vectors_tensor = torch.Tensor(english_vectors)
    english_vectors_tensor = make_tensor(english_vectors, unit)
    
    for index in range(len(french_test)):

        if len(french_test[index]) == 0:
            continue
        # for testing only
        # french_test[index] = ["l'atelier", 'aura', 'lieu', 'aux', 'dates', 'ci-après', 'lundi', 'novembre', 'de', 'à', 'heures', 'et', 'de', 'heures', 'novembre', 'de', 'heures', 'mercredi', 'novembre', 'de', 'à']
        # eng_test[index] = ['the', 'workshop', 'will', 'be', 'held', 'on', 'the', 'following', 'dates', 'monday', 'november', 'from', 'am', 'pm', 'from', 'pm', 'pm', 'tuesday', 'november', 'from', 'pm']

        print("Sentence {}".format(index))
        print("French sentence: {}".format(french_test[index]))
        print("English sentence: {}".format(eng_test[index]))

        translated_english_words = []

        # get first english original word
        first_english_original = eng_test[index][0]
        # get first french word
        first_french_word = french_test[index][0]
        first_french_word_index = french_vocab.index(first_french_word)
        # get predicted french words for all english words
        predictions = translation_model.predict(english_vectors_tensor)
        translated_word = None
        max_pred_score = -1
        # cycle through all predictions
        for i, pred in enumerate(predictions):
            # get score for this word's prediction from one hot output            
            pred_score = get_score_for_word(pred, first_french_word_index)
            if pred_score > max_pred_score:
                max_pred_score = pred_score
                # translated_vector = english_vectors[i]
                translated_word = eng_vocab[i]

        # get the corresponding english word of this vector
        # translated_word = get_predicted_word(translated_vector, eng_vocab)
        translated_english_words.append(translated_word)
        
        actual_translations.append(first_english_original)
        predicted_translations.append(translated_word)
        
        # the trigram now will be <start> and translated_word
        first_word = '<start>'
        second_word = translated_word
        # for word_index in french_test[index]:
        for word_index in range(1, len(french_test[index])):  # Looping from 1 to end of sentence, skipping index 0

            # print("Word {}, {}".format(word_index, french_test[index][word_index]))

            bigram = np.hstack((w2v_vectors[first_word], w2v_vectors[second_word]))
            #bigram = torch.Tensor([bigram])
            bigram = make_tensor([bigram], unit)
            
            next_word_pred = trigram_model.predict(bigram)
            
            top_50_prediction_indices = get_top_n_predictions(next_word_pred, n=50)

            next_french_word = french_test[index][word_index]
            next_french_word_index = french_vocab.index(next_french_word)
            
            max_pred_score = -1
            translated_word = None

            for i in top_50_prediction_indices:
                print(i)
                print(eng_vocab[i])
            # top_50_eng_words = [eng_vocab[i] for i in top_50_prediction_indices]
            top_50_eng_vectors = [w2v_vectors[w] for w in top_50_eng_words]
            #top_50_eng_vectors_tensor = torch.Tensor(top_50_eng_vectors)
            top_50_eng_vectors_tensor = make_tensor(top_50_eng_vectors, unit)

            translated_predictions = translation_model.predict(top_50_eng_vectors_tensor)
            for i, pred in enumerate(translated_predictions):
                score = get_score_for_word(pred, next_french_word_index)
                if score > max_pred_score:
                    max_pred_score = score
                    translated_word = top_50_eng_words[i]

            # for pred_index in top_50_prediction_indices:
            #     eng_word = eng_vocab[pred_index]
            #     eng_word_vector = w2v_vectors[eng_word]

            #     # Append vector to the list
            #     eng_word_vector = [eng_word_vector]

            #     # Make it a tensor out of the list
            #     vector_tensor = torch.Tensor(eng_word_vector)

            #     translated_pred = translation_model.predict(vector_tensor)  # was eng_word_vector
            #     translated_pred = translated_pred[0]
                    
            #     score = get_score_for_word(translated_pred, next_french_word_index)
            #     if score > max_pred_score:
            #         max_pred_score = score
            #         translated_word = eng_word

            translated_english_words.append(translated_word)
            actual_translations.append(eng_test[index][word_index])
            predicted_translations.append(translated_word)
            
            # calculates scores
            # total_translations += 1
            # if translated_word == next_english_original:
            #     correct_translations +=1

            first_word = second_word
            second_word = translated_word

        print("Translated sentence: {}".format(translated_english_words))

    print(classification_report(actual_translations, predicted_translations))


def test_new(eng_test, french_test, eng_vocab, french_vocab,w2v_vectors,eng_indices, fr_eng_indices, len_eng, len_fr, trigram_model, translation_model, unit, top_n):
    
    # Output variables
    # Measurements should be accuracy, precision, recall and F1_score
    actual_translations = []
    predicted_translations = []
    
    # create a list of w2v vectors of all english words in vocab
    english_vectors = []
    for word in eng_vocab:
        vect = w2v_vectors[word]
        english_vectors.append(vect)
    english_vectors_tensor = make_tensor(english_vectors, unit)
    
    for index in range(len(french_test)):

        if len(french_test[index]) == 0:
            continue

        print("Sentence {}".format(index))
        print("French sentence: {}".format(french_test[index]))
        print("English sentence: {}".format(eng_test[index]))

        translated_english_words = []

        # get first english original word
        first_english_original = eng_test[index][0]
        # get first french word
        first_french_word = french_test[index][0]
        first_french_word_index = french_vocab.index(first_french_word)
        # get predicted french words for all english words
        predictions = translation_model.predict(english_vectors_tensor)
        translated_word = None
        max_pred_score = -1
        # cycle through all predictions
        for i, pred in enumerate(predictions):
            # get score for this word's prediction from one hot output            
            pred_score = get_score_for_word(pred, first_french_word_index)
            if pred_score > max_pred_score:
                max_pred_score = pred_score
                translated_word = eng_vocab[i]

        # get the corresponding english word of this vector
        translated_english_words.append(translated_word)
        
        actual_translations.append(first_english_original)
        predicted_translations.append(translated_word)
        
        # the trigram now will be <start> and translated_word
        first_word = '<start>'
        second_word = translated_word
        for word_index in range(1, len(french_test[index])):  # Looping from 1 to end of sentence, skipping index 0

            bigram = np.hstack((w2v_vectors[first_word], w2v_vectors[second_word]))
            bigram = make_tensor([bigram], unit)            
            next_word_pred = trigram_model.predict(bigram)            
            top_50_prediction_indices = get_top_n_predictions(next_word_pred, top_n)
            next_french_word = french_test[index][word_index]
            next_french_word_index = french_vocab.index(next_french_word)
            
            max_pred_score = -1
            translated_word = None

            top_50_eng_words = [eng_vocab[i] for i in top_50_prediction_indices]
            top_50_eng_vectors = [w2v_vectors[w] for w in top_50_eng_words]

            top_50_eng_vectors_tensor = make_tensor(top_50_eng_vectors, unit)

            translated_predictions = translation_model.predict(top_50_eng_vectors_tensor)
            for i, pred in enumerate(translated_predictions):
                score = get_score_for_word(pred, next_french_word_index)
                if score > max_pred_score:
                    max_pred_score = score
                    translated_word = top_50_eng_words[i]

            translated_english_words.append(translated_word)
            actual_translations.append(eng_test[index][word_index])
            predicted_translations.append(translated_word)
            
            first_word = second_word
            second_word = translated_word

        print("Translated sentence: {}".format(translated_english_words))

    print(classification_report(actual_translations, predicted_translations))