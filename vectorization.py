# this file contains helper functions for generating word vectors
from sklearn.feature_extraction import DictVectorizer

import warnings # Stackoverflow said to do this if you use Windows
from gensim.models.keyedvectors import KeyedVectors
from trigrams import create_ngram
import numpy as np

warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')


def get_vocabulary(ls_text):
    vocabulary = []
    for ls_words in ls_text:
        vocabulary.extend(ls_words)
    vocabulary = list(set(vocabulary+["<start>"]))
    
    return vocabulary


def remove_words(english, french, w2v_model):

    print("Number of English sentences: {}".format(len(english)))
    
    e_data = []
    f_data = []
    not_found = []
    
    for i in range(len(english)):
        english_sentence = english[i]
        french_sentence = french[i]

        #Loop backwards so that no items are skipped
        for idx in reversed(range(len(english_sentence))):
            word = english_sentence[idx]

            try:
                vec = w2v_model.word_vec(word)
            except KeyError:
                english_sentence.pop(idx)
                french_sentence.pop(idx)
                if word not in not_found:
                    not_found.append(word)
                continue
                    
        e_data.append(english_sentence)
        f_data.append(french_sentence)

    print("Words not found in model: {}".format(not_found))
    return e_data, f_data

def generate_one_hot_encoded_vectors(ls_words):
    # encoder = OneHotEncoder(ls_words)
    # one_hot_encodings = encoder.fit_transform(ls_words)
    dv = DictVectorizer()
    dvX = dv.fit_transform( [ {'word': a} for a in ls_words ] )
    dvX = dvX.toarray()
    one_hot_encodings = {}
    for index in range(len(ls_words)):
        one_hot_encodings[ls_words[index]] = dvX[index]
    return one_hot_encodings


def load_gensim_model(path_to_model):
    model =  KeyedVectors.load_word2vec_format(path_to_model, binary=True)
    return model


def get_w2v_vectors(w2v_model, ls_words):
    w2v_vectors = {}
    for word in ls_words:
        try:
            vec = w2v_model.word_vec(word)
            w2v_vectors[word] = vec
        # this exception will occur when a word does not exist in the vocabulary of this model
        except KeyError:
            if word == '<start>':
                vec = np.random.rand(1,300)[0]
                w2v_vectors[word] = vec
    return w2v_vectors


def make_vector_trigrams(sentences, w2v_vectors):
    sentence_vector_trigrams = []
    missing_words = []

    trigram1 = sentences[0][0]
    
    for sentence in sentences:
        trigram_vectors = []
        for trigram in sentence:
            tg_vector = []
            for word in trigram:

                #Do we need this check now that we've taken care of this?
                if word not in w2v_vectors:
                    if word not in missing_words:
                        missing_words.append(word)
                else:
                    vector = w2v_vectors[word]
                    tg_vector.append(vector)

            trigram_vectors.append(tg_vector)
        sentence_vector_trigrams.append(trigram_vectors)

    #print("The following words were not found in the w2v: ")
    #print(missing_words)
    print("{} trigram vectors were created".format(len(trigram_vectors)))
    print("Vector for {}: {}".format(trigram1, trigram_vectors[0]))
    return sentence_vector_trigrams
