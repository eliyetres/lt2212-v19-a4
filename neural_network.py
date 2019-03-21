import torch
import torch.nn as nn


class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size) 
        self.sigmoid = nn.Sigmoid()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.sigmoid(out)
        out = self.fc2(out)
        return out


def train_model(model, X, Y, learning_rate, n_epochs):
    # initialize the loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # commence training
    for epoch in range(n_epochs):
        outputs = model(X)
        # loss = criterion(outputs, Y)
        loss = criterion(outputs, torch.max(Y, 1)[1])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print ('Epoch [{}/{}], Loss: {:.4f}' 
                   .format(epoch+1, n_epochs, loss.item()))

    return model