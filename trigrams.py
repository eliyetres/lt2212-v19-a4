# Creates trigrams
from nltk import ngrams


def create_ngram(vocabulary):
    """ Takes a list of word form the vocabulary.
    Returns trigrams as tuples in a list of lists (one for every sentence) padded with starting tag. """
    all_trigrams = []
    for sentence in vocabulary:
        trigram_sentence = []
        # Padding n-grams with start tag
        generated_grams = ngrams(
            sentence, 3, pad_left=True, left_pad_symbol='<start>', pad_right=False)
        for each_gram in generated_grams:
            trigram_sentence.append(each_gram)
        all_trigrams.append(trigram_sentence)
    return all_trigrams
