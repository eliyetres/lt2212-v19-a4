#Training and testing script
from neural_network import *
import torch
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
