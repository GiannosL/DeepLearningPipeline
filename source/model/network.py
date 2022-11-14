import torch
import pandas as pd
import torch.nn as nn
import matplotlib.pyplot as plt
from source.output.predictions import Predictions


class ANN(nn.Module):

    def __init__(self, input_layer_nodes, output_layer_nodes, hyper_params, name="Pythagoras"):

        super().__init__()
        # Model name
        self.name = name
        
        #
        self.hyper_parameters = hyper_params
        self.in_features = input_layer_nodes
        self.h1_nodes = self.hyper_parameters["n_units_1"]
        self.out_features = output_layer_nodes
        self.dropout_rate = self.hyper_parameters["dropout_rate"]

        #
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
    
    def train_model(self, X, y_true, work_directory):
        
        X = convert_to_tensor(X)
        y_true = convert_to_tensor(y_true, target=True)

        #
        self.training_set_size = X.shape[0]

        losses = []
        for epoch in range(self.epochs):
            #
            y_pred = self.model(X)

            #
            loss = self.loss_function(y_pred, y_true)
            losses.append(loss.item())

            # back-propagation
            self.model.zero_grad()
            loss.backward()
            self.optimizer.step()
        
        self.training_loss = loss.item()
        self.loss_history = losses
        self.plot_loss_history(work_directory)
    
    def setup_training(self, loss_func=nn.CrossEntropyLoss(), 
                       optimizer=torch.optim.Adam, epochs=1000,
                       learning_rate=0.01):
        
        self.learning_rate = learning_rate
        self.loss_function = loss_func
        self.optimizer = optimizer(self.model.parameters(), lr=self.learning_rate)
        self.epochs = epochs

    def setup_model(self):          
        #
        model = nn.Sequential(
            nn.Linear(self.in_features, self.h1_nodes),
            nn.ReLU(),
            nn.Linear(self.h1_nodes, self.out_features)
        )

        return model
    
    def predict(self, X, y):
        X = convert_to_tensor(X)
        y_pred = self.model(X)

        _, y_pred = torch.max(y_pred, 1)
        
        return Predictions(data=X, y_true=y, y_pred=y_pred)
    
    def plot_loss_history(self, work_dir):
        #
        plt.figure(figsize=(10, 7))
        #
        plt.plot(range(1, self.epochs+1), self.loss_history)
        # 
        plt.title("Loss trajectory over epochs", fontsize=16, weight="bold")
        plt.ylabel("Loss", fontsize=12)
        plt.xlabel("Epochs", fontsize=12)
        plt.ylim((0, 1))
        # save plot
        plt.savefig(f"{work_dir}results/plots/training_loss.png", dpi=1200)
        plt.clf()
    
    def save(self, name):
        torch.save(self.state_dict(), name)
        print(f"Model saved as: {name}.")


def convert_to_tensor(df: pd.DataFrame, target=False):
    if target:
        return torch.LongTensor(df["condition"].values)
    else:
        return torch.FloatTensor(df.values)
