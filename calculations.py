#Calculations.py


#Try this one
def cross_entropy(X,y):
    """
    X is on the form num_examples x num_classes
    y is labels (num_examples x 1)
    	Note that y is not one-hot encoded vector. 
    	It can be computed as y.argmax(axis=1) from one-hot encoded vectors of labels if required.
    """

    #If Y are one-hot encoded vectors
    y = y.argmax(axis=1)
    
    m = y.shape[0]
    p = softmax(X)

    # Multidimensional array indexing to extract 
    # softmax probability of the correct label for each sample.
    log_likelihood = -np.log(p[range(m),y])
    loss = np.sum(log_likelihood) / m
    
    return loss
