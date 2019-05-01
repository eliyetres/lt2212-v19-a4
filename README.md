# LT2212 V19 Assignment 4
Classic SMT and feed-forward neural networks\
From Asad Sayeed's statistical NLP course at the University of Gothenburg. 

## Group Rho ρ
Azfar Imtiaz\
Elin Hagman\
Sandra Derbring

## Running the program
The models are trained by running the script **train_model.py**.
### Arguments:
Targetfile: File for target language. (Default: UN-english.txt)\
Sourcefile: File for source language. (Default: UN-french.txt)\
Modelfile: Pre-trained vector file. (Default: GoogleNews-vectors-negative300.bin)\
-M: Model type. 0 for trigram model or 1 for the translation model. (Default 0)\
-B: Batch size for loading data into the model. A batch size of 1 is one sentence. (Default: 100)\
-E: Epochs used to train the model for every batch. (Default: 20)\
-L: Layer size for the neural network. (Default: 600)\
-P: Using CPU or GPU for training the model. Select GPU with e.g. "cuda:0". (Default CPU)\
-R: Learning rate for the Adam optimizer (Default: 0.01)
