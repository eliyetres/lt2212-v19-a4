# Report

### Preprocessing
The parallel corpora was preprocessed by lowercasing and word tokenization. We specified which characters would be removed instead of using shorthand regular expressions because many of the French characters would otherwise have been considered non-letters. 

### Vectorization
One-hot vectors for both languages and word2vec vectors for English were created with the help of the pre-trained Python gensim model. Out-of-vocabulary words were removed. The one-hot vectorization was initially done for all the words at once but this also produced memory errors because of the vector size when it was tested on the whole file.  It was changed to use sparse matrices instead of dense matrices for better memory management but it created more errors in the code that we didn't manage to solve. Using indices for words and switching to generating one-hot vectors with numpy in the translation model when needed, word for word, eventually solved the error.

### Implementing the models
We started by trained a basic network on the data using PyTorch's nn classes, just to get the basic flow working. Once that was up and running, we systematically started replacing parts of the training process with our own implementations, such as the matrix multiplications for training, categorical cross-entropy for calculating the loss, testing the model etc. The Adam optimizer was used to optimize over the weights.

When discussing the layer size with Bill he mentioned the size of the hidden nodes should be somewhere between the dimension of the input layer and the dimension of the output (target langauge vocabulary). We decided that around 500 would be a good start. Experimenting further, a layer size of 600 seemed to do better and produce a lower loss than 500 and took less time to train than a layer size of 800 where the loss was about the same. 

### Training and testing
To be able to reuse trained models, the training and testing is done in separate scripts. The training script processes either the translation model or the language model using parameters mentioned in the README. Various minor functions are implemented in separate scripts that is imported for the training and testing.

Initially all the data was loaded into the model at once which resulted in memory errors. We looked into using a data loader but it seemed hard to implement since we already had finished the loading and processing of the files. We ended up splitting the data into batches to feed into the model piece py piece. The batch number equals how many sentences that are loaded and used for training. When the model is created it uses one batch from the data to initiate the weight and then the rest is loaded one batch at a time. Each model is saved to a file after one batch of data has been processed. If the training is interrupted, it is then possible to resume training from the last saved model.

The standard batch size was set to a small value to make sure it didn't run into any memory issues and the epochs were also set to a lower value because of the time it would take to train the model.

When testing the models as a translation system, the trained models are loaded and the data is divided into a training set and a test set. The default size of the test data is 20% of the dataset. The prediction is done sentence by sentence. The first word in the sentence is always treated separately to be able to make a trigram for predicting the most probable next English words. 50 was used as a default parameter for the next top words. 

### Results
The script measures accuracy, precision, recall and F1 score with the help of SKLearn Metrics. However, we have not yet mananged to run data large enough to get these scores for the prediction. Below is the largest data set we have been able to train.

| Sentences | Trained on | Tested on | Trigram loss | Translation loss |
|-----------|------------|-----------|--------------|------------------|
| 75 000    | 60 000     | 15 000    | 6.3979       | 5.9562           |

