# Report

### Preprocessing
The parallel corpora was preprocessed by lowercasing and word tokenization. We specified which characters would be removed instead of using shorthand regular expressions because many of the French characters would otherwise have been considered non-letters. 

### Vectorization
One-hot vectors for both languages and word2vec vectors for English were created with the help of the pre-trained Python gensim model. Out-of-vocabulary words were removed. The one-hot vectorization was initially done for all the words at once but this also produced memory errors because of the vector size when it was tested on the whole file. It was changed to use sparse matrices instead of dense matrices for better memory management but it created more errors in the code that we didn't manage to solve. Using indices for words and switching to generating one-hot vectors with numpy in the translation model when needed, word for word, eventually solved the error.

### Implementing the models
We started by trained a basic network on the data using PyTorch's nn classes, just to get the basic flow working. Once that was up and running, we systematically started replacing parts of the training process with our own implementations, such as the matrix multiplications for training, categorical cross-entropy for calculating the loss, testing the model etc. The Adam optimizer was used to optimize over the weights.

When discussing the layer size with Bill he mentioned the size of the hidden nodes should be somewhere between the dimension of the input layer and the dimension of the output (target langauge vocabulary). We decided that around 500 would be a good start. Experimenting further, a layer size of 600 seemed to do better and produce a lower loss than 500 and took less time to train than a layer size of 800 where the loss was about the same. 

### Training and testing
To be able to reuse trained models, the training and testing is done in separate scripts. The training script trains either the translation model or the language model using parameters mentioned in the README. This is to ensure that the models can be trained simultaneously. Various minor functions are implemented in separate scripts that are imported for the training and testing. Sanity checks to make sure the arguments are valid have been implemented for each argument in both the training and testing script. 

Initially all the data was loaded into the model at once which resulted in memory errors. We looked into using a data loader but it seemed hard to implement since we already had finished the loading and processing of the files. We ended up splitting the data into batches to feed into the model piece py piece. The batch number equals how many sentences that are loaded and used for training. When the model is created it uses one batch from the data to initiate the weight and then the rest is loaded one batch at a time. Each model is saved to a file after one batch of data has been processed. If the training is interrupted, it is then possible to resume training from the last saved model.

The standard batch size was set to a small value to make sure it didn't run into any memory issues and the epochs were also set to a lower value because of the time it would take to train the model.

When testing the models as a translation system, the trained models are loaded and the data is divided into a training set and a test set. The default size of the test data is, by convention, 20% of the dataset but the user can specify another percentage when running the program. The prediction is done sentence by sentence. The first word in the sentence is always treated separately to be able to make a trigram for predicting the most probable next English words. 50 was used as a default parameter for the next top words but it can be chosen by the user. 

## Bonus A: GPU
It's possible to train the model on the GPU sing the parameter "cuda:0" (or any other number of available GPU). We use a function that either creates standard tensors for CPU or cuda tensors for GPU. If the GPU is selected, we push the tensors to it using the function to.(device) where device is the selected GPU. 

Using the GPU was very efficent in the beginning when we were training the model without splitting the data into batches, since we were only feeding the data into the model once.
The speed of the GPU is only utilized when training the model during epochs, so a small batch size and a low number or epochs might not have an effect on performance since it would require time to save the data to the disk and move the data from CPU to GPU.

### Results
Although we have implemented a complete neural network, it is, due to obvious reasons, both inefficient and unoptimized. This makes it both slow and memory-consuming and caused a lot of errors along the way. Those obstructions, together with limited time frame, made it difficult for us to experiment too much with different parameters. 


From our experience we can conclude a few things:
* The layer size equals the number of nodes in the hidden network, and a larger size means calculations take longer.
* The more epochs, the longer it will train on the same set of data and the loss will decrease.
* A higher batch size means you also might need to increase epochs so it trains and produces a reasonable low loss.

#### Training speed
We ran some speed test when training the language model shown in the table. The results show that the GPU speed is best utilized when loading a larger batch and training the model for a longer amount of time.


| Sentences | Batch size | Epochs | CPU           | GPU               | Comment                    |                                                                                                                                                     
|-----------|------------|--------|---------------|-------------------|----------------------------|
| 16000     | 200        | 50     | 25 min 29 sec | 25 min 17 seconds | No real time difference. In this case, using CPU was 8 sec faster.                                                                                                                |
| 16000     | 200        | 100    | ? (+20 min)   | 9 min 27 sec      | Training on the GPU finished after 9 min 27 sec, at this point the training done on CPU had not even finished the first batch. After 20 min of waiting we interrupted the script. |
|           |            |        |               |                   |                                                                                                                                                                                   |

#### Translation metrics

The script measures accuracy, precision, recall and F1 score with the help of SKLearn Metrics. However, we have not yet managed to run data large enough to get these scores for the prediction. Below is the largest data set we have been able to train, due to the aforementioned memory errors and time frames.


| Sentences | Trained on | Tested on | Trigram loss | Translation loss |
|-----------|------------|-----------|--------------|------------------|
| 75 000    | 60 000     | 15 000    | 6.3979       | 5.9562           |

