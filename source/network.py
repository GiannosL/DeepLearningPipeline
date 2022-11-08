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
        self.out_features = out_features

        # layers: input=4 -> h1 -> h2 N --> output=3
        # input - h1
        self.fc1 = nn.Linear(in_features, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.fc3 = nn.Linear(h2, h3)
        self.out = nn.Linear(h3, out_features)
    
    def model_description(self):
        print(f"Input layer:\t{self.in_features}")
        print(f"Output layer:\t{self.out_features}")
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.out(x)

        return x
    
    def save(self, name):
        torch.save(self.state_dict(), name)
        print(f"Model saved as: {name}.")
