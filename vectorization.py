# this file contains helper functions for generating word vectors
from sklearn.feature_extraction import DictVectorizer

import warnings # Stackoverflow said to do this if you use Windows
from gensim.models.keyedvectors import KeyedVectors

warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')


def get_vocabulary(ls_text):
    vocabulary = []
    for ls_words in ls_text:
        vocabulary.extend(ls_words)
    vocabulary = list(set(vocabulary))
    return vocabulary


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
            continue
    return w2v_vectors
