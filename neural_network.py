import torch
from torch.nn import CrossEntropyLoss
# import torch.nn as nn
import random


# class NeuralNetwork(nn.Module):
#     def __init__(self, input_size, hidden_size, num_classes):
#         super(NeuralNetwork, self).__init__()
#         self.fc1 = nn.Linear(input_size, hidden_size) 
#         self.sigmoid = nn.Sigmoid()
#         self.fc2 = nn.Linear(hidden_size, num_classes)

#     def forward(self, x):
#         out = self.fc1(x)
#         out = self.sigmoid(out)
#         out = self.fc2(out)
#         return out


# def train_model(model, X, Y, learning_rate, n_epochs):
#     # initialize the loss and optimizer
#     criterion = nn.CrossEntropyLoss()
#     optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

#     # commence training
#     for epoch in range(n_epochs):
#         outputs = model(X)
#         # loss = criterion(outputs, Y)
#         loss = criterion(outputs, torch.max(Y, 1)[1])

#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#         print ('Epoch [{}/{}], Loss: {:.4f}' 
#                    .format(epoch+1, n_epochs, loss.item()))

#     return model


class NeuralNetwork():
    def __init__(self, device, lr=0.01):
        self.learning_rate = lr
        if device == "gpu":
            self.device = torch.device("cuda:0")
            print("Using GPU")              
        else: 
            self.device = torch.device("cpu")  
            print("Using CPU")

    
    def forward(self, X):
        # d = self.weights_1.dot(x)
        # d = torch.sigmoid(d + self.bias_1)
        # d = d.dot(self.weights_2)

        # One hidden layer
        #d = X.mm(self.weights_1)
        #d = torch.sigmoid(d + self.bias_1)
        #d = d.mm(self.weights_2)
        #self.output = torch.softmax(d + self.bias_2, dim=1)

        # 2 hidden layers
        d = X.mm(self.weights_1)
        d = torch.sigmoid(d + self.bias_1)
        d = d.mm(self.weights_2)
        d = torch.sigmoid(d + self.bias_2)
        d = d.mm(self.weights_3)
        # use log_softmax to avoid vanishing gradient problem
        self.output = torch.log_softmax(d + self.bias_3, dim=1)

        
    def train(self, X, Y, hidden_size, num_classes, n_epochs=20):
        print(self.device)
        input_feature_size = len(X[0])        
        #self.weights_1 = torch.zeros((input_feature_size, hidden_size), requires_grad=True)
        #self.weights_2 = torch.zeros((hidden_size, hidden_size), requires_grad=True)
        #2 hidden layers
        #self.weights_3 = torch.zeros((hidden_size, num_classes), requires_grad=True)
        # self.weights_1 = torch.randn(input_feature_size, requires_grad=True)
        # self.weights_2 = torch.randn(num_classes, requires_grad=True)
        
        #self.bias_1 = torch.zeros(1, requires_grad=True)
        #self.bias_2 = torch.zeros(1, requires_grad=True)
        # 2 hidden layers
        #self.bias_3 = torch.zeros(1, requires_grad=True)      

        # using 2 hidden layers
        #self.weights_1 = torch.randn((input_feature_size, hidden_size), requires_grad=True)
        #self.weights_2 = torch.randn((hidden_size, hidden_size), requires_grad=True)        
        #self.weights_3 = torch.randn((hidden_size, num_classes), requires_grad=True)

        #self.bias_1 = torch.randn(1, requires_grad=True)
        #self.bias_2 = torch.randn(1, requires_grad=True)
        #self.bias_3 = torch.randn(1, requires_grad=True)

        # 2 hidden layers with GPU option
        self.weights_1 = torch.randn((input_feature_size, hidden_size),requires_grad=True, device=self.device)    
        self.weights_2 = torch.randn((hidden_size, hidden_size),  requires_grad=True, device=self.device)
        self.weights_3 = torch.randn((hidden_size, num_classes), requires_grad=True, device=self.device) 

        self.bias_1 = torch.randn(1, requires_grad=True, device=self.device)
        self.bias_2 = torch.randn(1, requires_grad=True, device=self.device)
        self.bias_3 = torch.randn(1, requires_grad=True, device=self.device)

        self.weights_1 = self.weights_1.to(self.device)
        self.weights_2 = self.weights_2.to(self.device)
        self.weights_3 = self.weights_3.to(self.device)

        self.bias_1 = self.bias_1.to(self.device)
        self.bias_2 = self.bias_2.to(self.device)
        self.bias_3 = self.bias_3.to(self.device)


        # initialize the optimizer
        #optimizer = torch.optim.Adam([self.weights_1, self.bias_1, self.weights_2, self.bias_2], lr=self.learning_rate)
        optimizer = torch.optim.Adam([self.weights_1, self.bias_1, self.weights_2, self.bias_2, self.weights_3, self.bias_3], lr=self.learning_rate)
        
        criterion = CrossEntropyLoss()

        # X_in = X[torch.randperm(X.size()[0])]
        X_in = X
        for epoch in range(n_epochs):
            print("Starting epoch{}".format(epoch))
            # for i in range(len(X)):

            nr_of_batches = len(X_in)/128
            batchsize = 128
            batch = 0
            
            for i in range(0, len(X_in), batchsize):
                  print("Starting batch {} of {} for epoch {}".format(batch, nr_of_batches, epoch))
                  x_batch = X_in[i:i+batchsize]
                  y_batch = Y[i:i+batchsize]
                  X_tensor, Y_tensor = self.make_tensor(x_batch, y_batch)
                  self.forward(X_tensor)
                  optimizer.zero_grad()
                  loss = self.cross_entropy_cat(self.output, torch.max(Y_tensor, 1)[1])
                  loss.backward()
                  optimizer.step()
                  print("\n")
                  batch +=1

            # do the forward pass
            #self.forward(X_in)

            # set the gradients to 0 before backpropagation
            #optimizer.zero_grad()

            # compute the loss - not used
            # loss = Y - self.output

            #Computing loss by built-in function - not used
            #loss = criterion(self.output, torch.max(Y, 1)[1])

            #Compute loss by own function - used
            #loss = self.cross_entropy_cat(self.output, torch.max(Y, 1)[1])

            # compute gradients
            #loss.backward()

            # update weights
            #optimizer.step()

            # X_in = X[torch.randperm(X.size()[0])]

            print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, n_epochs, loss.item()))
            #print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, n_epochs, loss))

        #random.shuffle(epoch) # shuffle the data might be useful if we decide to use batch sizes


    def cross_entropy_cat(self, x,y, epsil=1e-12):
                                                                                                                                                                            
        #X is on the form num_examples x num_classes                                                                                                                                
        #y is labels (num_examples x 1)                                                                                                                                             
        #Note that y is not one-hot encoded vector.                                                                                                                                 
        #It can be computed as y.argmax(axis=1) from one-hot encoded vectors of labels if required.                                                                                 
        
        #If Y are one-hot encoded vectors                                                                                                                                          
        #y = np.argmax(y, axis=1)                                                                                                                                                       
        m = y.shape[0]
        p = torch.softmax(x+epsil, dim=1)                                                                                                                                            

        # Multidimensional array indexing to extract
        # softmax probability of the correct label for each sample.
        log_likelihood = -torch.log(p[range(m),y]+epsil)    # added epsil to avoid log(0)
        loss = torch.sum(log_likelihood)/m

        return loss


    def predict(self, X):

        # One hidden layer                                                                                                                                                          
        #d = X.mm(self.weights_1)                                                                                                                                                   
        #d = torch.sigmoid(d + self.bias_1)                                                                                                                                         
        #d = d.mm(self.weights_2)                                                                                                                                                   
        #self.output = torch.softmax(d + self.bias_2, dim=1)                                                                                                                        

        # 2 hidden layers                                                                                                                                                           
        d = X.mm(self.weights_1)
        d = torch.sigmoid(d + self.bias_1)
        d = d.mm(self.weights_2)
        d = torch.sigmoid(d + self.bias_2)
        d = d.mm(self.weights_3)

        #Use softmax, not log_softmax
        predicted = torch.softmax(d + self.bias_3, dim=1)
        
        return predicted

    
    def make_tensor(self, X_list, Y_list):

        if self.device == "cuda:0":
            # Using GPU (fast)                                                                                                                                                      
            X = torch.cuda.FloatTensor(X_list) # gpu variable must have input type FloatTensor                                                                                      
            Y = torch.cuda.FloatTensor(Y_list)
        else:
        # Using CPU (slow)                                                                                                                                                    
            X = torch.Tensor(X_list)
            Y = torch.Tensor(Y_list)

        return X, Y
