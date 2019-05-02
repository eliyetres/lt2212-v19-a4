#Testing script
from neural_network import NeuralNetwork
import torch
import operator
import numpy as np
from utils import create_ngram
from sklearn.metrics import classification_report


def make_tensor(vector_list, unit):
    if unit == "cpu":
        # Using CPU (slow)                                                                                                                                                          
        X = torch.Tensor(vector_list)
    else:
        # Using GPU (fast)                                                                                                                                                          
        X = torch.cuda.FloatTensor(vector_list) # gpu variable must have input type FloatTensor                                                                                            
    return X


def get_score_for_word(pred, first_french_word_index):
    return pred[first_french_word_index]


def get_top_n_predictions(next_word_pred, n=50):
    #sorted_preds = sorted(next_word_pred, reverse=False)

    # next_word_pred is a Tensor
    # use sorting method of torch, returning indices
    _, indices = torch.sort(next_word_pred, descending=True)

    top_n_indices = []
    
    for i in range(0, n):
        index = indices[0][i]
        top_n_indices.append(index)

    return top_n_indices


def test_translation(eng_test, french_test, eng_vocab, french_vocab,w2v_vectors,eng_indices, fr_eng_indices, trigram_model, translation_model, unit, top_n):
    
    # Output variables
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

        print("Sentence {} / {}".format(index, len(french_test)))
        print("French sentence: {}".format(' '.join(french_test[index])))
        
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
        
        # Looping from 1 to end of sentence, skipping index 0
        for word_index in range(1, len(french_test[index])):

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

        print("Translated sentence: {}".format(' '.join(translated_english_words)))
        print("\n")

    print(classification_report(actual_translations, predicted_translations))
