# Description of team member contribution

## Group ρ

### Azfar Imtiaz
* Vectorization, one-hot encoding of vocabulary
* Vectorization, load word vectors from Gensim
* Design a basic working network with PyTorch nn module
* Build the forward function with biases and weights
* Use sparse matrices instead of dense matrices for combating memory issues
* Clean up the code, create script for utility functions and remove extra scripts
* Convert the two separate language and translation model scripts into single scripts with parameter to specify model type

### Elin Hagman

* Open and reading text files, preprocessing text and tokenization.
* Building trigrams from vobabulary sentences.
* Added options for GPU mode
* Added zero grad before back propagation
* Added log_softmax to help with vanishing gradients
* Loading data in batches
* Creating scripts for running training and testing in terminal

### Sandra Derbring

* Splitting vocabulary into training and test data sets by percentage.
* Removing words from data that are not found in the model
* Building trigrams of vectors
* Writing cross entropy functionality
* Added predict function to the model
* Initializing training and testing scripts
* Saving models to files with possibility to resume training if interrupted
