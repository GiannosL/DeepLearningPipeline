import torch
import torch.nn as nn
import torch.nn.functional as F


class ANN(nn.Module):

    def __init__(self, 
                 in_features=4, 
                 h1 = 12,
                 h2 = 12,
                 h3 = 12,
                 out_features=3):

        super().__init__()
        #
        self.in_features = in_features
        self.h1 = h1
        self.h2 = h2
        self.h3 = h3
        self.out_features = out_features

        # layers: input=4 -> h1 -> h2 N --> output=3
        # input - h1
        self.fc1 = nn.Linear(in_features, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.fc3 = nn.Linear(h2, h3)
        self.out = nn.Linear(h3, out_features)
    
    def model_description(self):
        print("\n-------------------------------------------------------------------")
        print("Multi-layer perceptron model.")
        print("Total number of layers = 5")
        print(f"Input layer:\t{self.in_features} nodes")
        print(f"Hidden layer 1:\t{self.h1} nodes")
        print(f"Hidden layer 2:\t{self.h2} nodes")
        print(f"Hidden layer 3:\t{self.h3} nodes")
        print(f"Output layer:\t{self.out_features} nodes")

        print("All layers are using the Rectified Linear Unit activation function.")
        print("-------------------------------------------------------------------\n")
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.out(x)

        return x
    
    def save(self, name):
        torch.save(self.state_dict(), name)
        print(f"Model saved as: {name}.")
