# this file contains helper functions for generating word vectors
from sklearn.feature_extraction import DictVectorizer


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
    one_hot_encodings = dvX.toarray()
    return one_hot_encodings