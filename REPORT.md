# Report
## Program design
A script trains either the translation model or the language model using parameters mentioned in the README.\
Initially all the data was loaded into the model at once which resulted in memory errors. We looked into using a data loader but it seemed hard to implement since we already had finished the loading and processing of the files. We ended up splitting the data into batches to feed into the model piece py piece. The batch number equals how many sentences that are loaded and used for training. When the model is created it uses one batch from the data to set to initiate the weight and then the rest is loaded one batch at a time. Each model is saved to a file after one batch of data has been processed. If the training is interupted, the part of the model that has been trained is saved and it's possible to resume training from the point the training was interupted.\
\
 The standard batch size was set to a small value to make sure it didn't run into any memory issues and the epochs were also set to a lower value because of the time it would take to train the model. \
\
When discussing the layer size with Bill he mentioned the size of the hidden nodes should be somewhere between the dimension of the input layer (300) and the dimension of the output (target langauge vocabulary, 1000). We decided that around 500 would be a good start. A layer size of 600 seemed to do better and produce a lower loss than 500, and \
\
The one-hot vectorization was initially done for all the words at once but this also produced memory errors because of the vector size when it was tested on the whole file.  It was changed to use sparse matrices instead of dense matrices for better memory management but it created more errors in the code that we didn't manage to solve. Switching to generating one-hot vectors using numpy in the translation model when needed, word for word, eventually solved the error.

## Results
