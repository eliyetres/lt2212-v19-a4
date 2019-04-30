import torch
from torch.nn import CrossEntropyLoss
import random
import numpy as np


class NeuralNetwork():
    def __init__(self, device, lr=0.01):
        self.learning_rate = lr
        self.device = torch.device(device)
    
    def forward(self, X):
        # One hidden layer
        d = X.mm(self.weights_1)
        d = torch.sigmoid(d + self.bias_1)
        d = d.mm(self.weights_2)
        #self.output = torch.softmax(d + self.bias_2, dim=1)
        # use log_softmax to avoid vanishing gradient problem
        self.output = torch.log_softmax(d + self.bias_2, dim=1)
        
    def start(self, X, Y, hidden_size, num_classes):
        input_feature_size = len(X[0])        

        # 1 or 2 hidden layers with GPU option
        self.weights_1 = torch.randn((input_feature_size, hidden_size),requires_grad=True, device=self.device)    
        self.weights_2 = torch.randn((hidden_size, num_classes),  requires_grad=True, device=self.device)

        self.bias_1 = torch.randn(1, requires_grad=True, device=self.device)
        self.bias_2 = torch.randn(1, requires_grad=True, device=self.device)
        
        self.weights_1 = self.weights_1.to(self.device)
        self.weights_2 = self.weights_2.to(self.device)
        
        self.bias_1 = self.bias_1.to(self.device)
        self.bias_2 = self.bias_2.to(self.device)
        
        # initialize the optimizer
        self.optimizer = torch.optim.Adam([self.weights_1, self.bias_1, self.weights_2, self.bias_2], lr=self.learning_rate)
        
    def train(self, X_batch, Y_batch, n_epochs=20):
        print("Training batch...")
        for epoch in range(n_epochs):
            
            X_batch, Y_batch = self.make_tensor(X_batch, Y_batch)
            X_batch = X_batch.to(self.device)
            Y_batch = Y_batch.to(self.device)

            # do the forward pass
            self.forward(X_batch)

            # set the gradients to 0 before backpropagation
            self.optimizer.zero_grad()

            # Compute loss by own function - used
            loss = self.cross_entropy_cat(self.output, torch.max(Y_batch, 1)[1])

            # compute gradients
            loss.backward()

            # update weights
            self.optimizer.step()
            torch.cuda.empty_cache()
            #print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, n_epochs, loss.item()))
            
    def cross_entropy_cat(self, x,y, epsil=1e-12):
                                                                                                                                                                            
        ''' X is on the form num_examples x num_classes                                                                                                                                
            y is labels (num_examples x 1)                                                                                                                                             
            Note that y is not one-hot encoded vector.                                                                                                                                 
            It can be computed as y.argmax(axis=1) from one-hot encoded vectors of labels if required.'''
        
        #If Y are one-hot encoded vectors                                                                                                                                          
        #y = np.argmax(y, axis=1)                                                                                                                                                       
        m = y.shape[0]
        p = torch.softmax(x+epsil, dim=1)                                                                                                                                            

        # Multidimensional array indexing to extract softmax probability of the correct label for each sample.
        log_likelihood = -torch.log(p[range(m),y]+epsil)    # added epsil to avoid log(0)
        loss = torch.sum(log_likelihood)/m
        return loss

    def predict(self, X):
        # One hidden layer                                                                                                                                                          
        d = X.mm(self.weights_1)                                                                                                                                                   
        d = torch.sigmoid(d + self.bias_1)                                                                                                                                         
        d = d.mm(self.weights_2)
        #Use softmax, not log_softmax
        self.output = torch.softmax(d + self.bias_2, dim=1)  
        predicted = torch.softmax(d + self.bias_2, dim=1)
        return predicted
    
    def make_tensor(self, X_list, Y_list):
        if self.device == "cpu":
            # Using CPU (slow)  
            X = torch.Tensor(X_list)
            Y = torch.Tensor(Y_list)
        else:
            # Using GPU (fast)                                                                                                                                                      
            X = torch.as_tensor(X_list, dtype=torch.float, device=self.device)
            Y = torch.as_tensor(Y_list, dtype=torch.float, device=self.device)
        return X, Y
