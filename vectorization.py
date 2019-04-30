# this file contains helper functions for generating word vectors
from sklearn.feature_extraction import DictVectorizer

from gensim.models.keyedvectors import KeyedVectors
from trigrams import create_ngram
import numpy as np
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')


def generate_one_hot(indices, word):
    one_hot_vector = np.zeros(len(indices), dtype=np.float32)
    ind = indices[word]
    one_hot_vector[ind] = 1

    return one_hot_vector


def generate_indices(vocabulary):
    """ Returns words and their indices and the length of the vocab to be used for one-hot vectors """ 
    indices = {}
    for en, word in enumerate(vocabulary):
        indices[word] = en

    return indices


def generate_trigram_vector(trigram_sentences, w2v_vectors, indices):
    sentence_vector_trigrams = []

    for sentence in trigram_sentences:
        trigram_vectors = []
        for trigram in sentence:
            tg_vector = []
            
            for index, word in enumerate(trigram):
                if index < 2:
                    tg_vector.append(w2v_vectors[word])
                else:                  
                    tg_vector.append(generate_one_hot(indices, word))
            trigram_vectors.append(tg_vector)
        sentence_vector_trigrams.append(trigram_vectors)

    return sentence_vector_trigrams


def generate_w2v_vector(model, word):
    if word == '<start>':
        vector = np.random.rand(1,300)[0]
    else:
        vector = w2v_model.word_vec(word)
            
    return vector


def gen_tri_vec_split(trigram_sentences, w2v_vectors, indices):
    X = []
    Y = []

    for sentence in trigram_sentences:
        for trigram in sentence:
            tg_vector = []
            
            for index, word in enumerate(trigram):
                if index < 2:
                    tg_vector.append(w2v_vectors[word])
                else:                  
                    tg_vector.append(generate_one_hot(indices, word))
            X.append(np.hstack((tg_vector[0], tg_vector[1])))
            Y.append(tg_vector[-1])
    return X, Y


def generate_translation_vectors(eng_sents, french_sents, w2v_vectors, indices):
    X = []
    Y = []
    for sent_index in range(len(eng_sents)):
        # get the english and french sentences
        eng_sent = eng_sents[sent_index]
        french_sent = french_sents[sent_index]
        for w_index in range(len(eng_sent)):
            # get the english and french word
            eng_word = eng_sent[w_index]
            french_word = french_sent[w_index]
            # get the english (w2v) and french (one hot) word vectors
            eng_word_vector = w2v_vectors[eng_word]

            # french_word_vector = one_hot_encoded_vectors_french[french_word]
            french_word_vector = generate_one_hot(indices, french_word)

            # english word vector is input, french word vector is output
            X.append(eng_word_vector)
            Y.append(french_word_vector)
           
    return X, Y


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

    #print("Words not found in model: {}".format(not_found))
    return e_data, f_data


def generate_one_hot_encoded_vectors(ls_words):
    dv = DictVectorizer()
    dvX = dv.fit_transform( [ {'word': a} for a in ls_words ] )
    # dvX = dvX.toarray()
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


def make_vector_trigrams(sentences, w2v_vectors, one_hot_vectors):
    sentence_vector_trigrams = []
    missing_words = []

    for sentence in sentences:
        trigram_vectors = []
        for trigram in sentence:
            tg_vector = []
            
            for index, word in enumerate(trigram):
                if index < 2:
                    vector = w2v_vectors[word]
                    tg_vector.append(vector)
                else:
                    vector = one_hot_vectors[word]
                    tg_vector.append(vector)

            trigram_vectors.append(tg_vector)
        sentence_vector_trigrams.append(trigram_vectors)

    #print("The following words were not found in the w2v: ")
    #print(missing_words)
    return sentence_vector_trigrams


def make_translation_vectors(eng_sents, french_sents, w2v_vectors, one_hot_encoded_vectors_french):
    translation_vectors_X = []
    translation_vectors_Y = []
    for sent_index in range(len(eng_sents)):
        # get the english and french sentences
        eng_sent = eng_sents[sent_index]
        french_sent = french_sents[sent_index]
        for w_index in range(len(eng_sent)):
            # get the english and french word
            eng_word = eng_sent[w_index]
            french_word = french_sent[w_index]
            # get the english (w2v) and french (one hot) word vectors
            eng_word_vector = w2v_vectors[eng_word]
            french_word_vector = one_hot_encoded_vectors_french[french_word]
            # english word vector is input, french word vector is output
            translation_vectors_X.append(eng_word_vector)
            translation_vectors_Y.append(french_word_vector)

    return translation_vectors_X, translation_vectors_Y
