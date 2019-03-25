#Training and testing script
from neural_network import *

def training(X, Y, model, num_of_words):

    #Variables for input
    hidden_layer_size = 600
    epochs = 4000
    num_of_classes = len(Y[0])
    #X is a list of w2v (one for each sentence)
    #Y is a list of labels in the form of one hot vectors (one for each sentence)

    #Output variables
    accuracy = 0
    precision = 0
    recall = 0
    F1_score = 0
    
    #For each sentence
    for sentence in X:

        #For the third word in the sentence:

        #Look up the probability distrubution
        #Run the translation model
        #Run the language model
        #Find the word of the argmax of the factor of these two

        #For the rest of the words in the sentence:
        for i in range(len(sentence)[3:]):
            #Use the start symbol and the word found above
            #Predict the num_of_words next English words

            #Use those words above to maximize the translation model probability
            #for the next French word



        
    #Calculate and print output measurements here
    #Also print out the translations
