import torch
import torch.nn as nn
import torch.nn.functional as F


class ANN(nn.Module):

    def __init__(self, name="Pythagoras"):

        super().__init__()
        # Model name
        self.name = name

        #
        self.setup_training()

        # model
        self.model = self.setup_model()
    
    def model_description(self):
        print("\n-------------------------------------------------------------------")
        print("Multi-layer perceptron model.")
        print("Total number of layers = 5")
        print(f"Input layer:\t{self.in_features} nodes")
        print(f"Hidden layer 1:\t{self.h1_nodes} nodes")
        print(f"Hidden layer 2:\t{'-'} nodes")
        print(f"Hidden layer 3:\t{'-'} nodes")
        print(f"Output layer:\t{self.out_features} nodes")

        print("All layers are using the Rectified Linear Unit activation function.")
        print("-------------------------------------------------------------------\n")
    
    def train_model(self, X, y_true):
        losses = []
        for epoch in self.epochs:
            #
            y_pred = self.model(X)

            #
            loss = self.loss_function(y_pred, y_true)
            losses.append(loss)

            #
            self.model.zero_grad()
            loss.backward()

            #
            self.optimizer.step()
    
    def setup_training(self, loss_func=nn.MSELoss(), 
                       optimizer=torch.optim.SGD(), epochs=1000):
        self.loss_function = loss_func
        self.optimizer = optimizer
        self.epochs = epochs

    def setup_model(self, input_layer_nodes, hidden_layer_nodes, output_layer_nodes, dropout, learning_rate):        
        #
        self.in_features = input_layer_nodes
        self.h1_nodes = hidden_layer_nodes
        self.out_features = output_layer_nodes
        
        #
        model = nn.Sequential(
            nn.Linear(self.in_features, self.h1_nodes),
            nn.ReLU(),
            nn.Linear(self.h1_nodes, self.out_features)
        )

        return model
    
    def predict(self, X):
        with torch.no_grad:
            y_pred = self.model(X)
        
        return y_pred
    
    def save(self, name):
        torch.save(self.state_dict(), name)
        print(f"Model saved as: {name}.")
