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
    def __init__(self, lr=0.01):
        self.learning_rate = lr

    def forward(self, X):
        # d = self.weights_1.dot(x)
        # d = torch.sigmoid(d + self.bias_1)
        # d = d.dot(self.weights_2)
        d = X.mm(self.weights_1)
        d = torch.sigmoid(d + self.bias_1)
        d = d.mm(self.weights_2)
        self.output = torch.softmax(d + self.bias_2, dim=1)

    def train(self, X, Y, hidden_size, num_classes, n_epochs=20):
        input_feature_size = len(X[0])
        self.weights_1 = torch.zeros((input_feature_size, hidden_size), requires_grad=True)
        self.weights_2 = torch.zeros((hidden_size, num_classes), requires_grad=True)
        # self.weights_1 = torch.randn(input_feature_size, requires_grad=True)
        # self.weights_2 = torch.randn(num_classes, requires_grad=True)
        self.bias_1 = torch.zeros(1, requires_grad=True)
        self.bias_2 = torch.zeros(1, requires_grad=True)

        # initialize the optimizer
        optimizer = torch.optim.Adam([self.weights_1, self.bias_1, self.weights_2, self.bias_2], lr=self.learning_rate)
        criterion = CrossEntropyLoss()

        # X_in = X[torch.randperm(X.size()[0])]
        X_in = X
        for epoch in range(n_epochs):
            # for i in range(len(X)):
            # do the forward pass
            self.forward(X_in)

            # compute the loss
            # loss = Y - self.output
            loss = criterion(self.output, torch.max(Y, 1)[1])
            # compute gradients
            loss.backward()

            # update weights
            optimizer.step()

            # X_in = X[torch.randperm(X.size()[0])]

            print ('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, n_epochs, loss.item()))