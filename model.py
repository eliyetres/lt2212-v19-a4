# Training model

import config
import warnings # Stackoverflow said to do this if you use Windows
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
# pip install gensim needed!
import gensim
from gensim.models import Word2Vec
from gensim.models.keyedvectors import KeyedVectors

model =  KeyedVectors.load_word2vec_format(config.google_model, binary=True)

# Print this to see if model works
#test = model.most_similar(positive=['woman', 'king'], negative=['man'])
#print(test)