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
        self.h1_nodes = self.hyper_parameters["n_units_1"] if self.hyper_parameters["n_layers"]>=1 else "-"
        self.h2_nodes = self.hyper_parameters["n_units_2"] if self.hyper_parameters["n_layers"]>=2 else "-"
        self.h3_nodes = self.hyper_parameters["n_units_3"] if self.hyper_parameters["n_layers"]==3 else "-"
        self.out_features = output_layer_nodes

        self.dropout_rate = self.hyper_parameters["dropout_rate"]
        self.layers = [
            self.in_features,
            self.hyper_parameters["n_units_1"], 
            self.hyper_parameters["n_units_2"], 
            self.hyper_parameters["n_units_3"]
            ]

        #
        self.model = self.setup_model()
    
    def model_description(self):
        print("\n-------------------------------------------------------------------")
        print("Multi-layer perceptron model.")
        print("Total number of layers = 5")
        print(f"Input layer:\t{self.in_features} nodes")
        print(f"Hidden layer 1:\t{self.h1_nodes} nodes")
        print(f"Hidden layer 2:\t{self.h2_nodes} nodes")
        print(f"Hidden layer 3:\t{self.h3_nodes} nodes")
        print(f"Output layer:\t{self.out_features} nodes")

        print("All layers are using the Rectified Linear Unit activation function.")
        print("-------------------------------------------------------------------\n")
    
    def train_model(self, X, y_true, work_directory):
        
        # pandas dataframe -> pytorch tensors
        X = convert_to_tensor(X)
        y_true = convert_to_tensor(y_true, target=True)

        # size of the training set
        self.training_set_size = X.shape[0]

        losses = []
        for epoch in range(self.epochs):
            # make prediction
            y_pred = self.model(X)

            # evaluate model's performance using the loss criterion
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
        """provide hyper-parameters for training the model"""
        self.learning_rate = learning_rate
        self.loss_function = loss_func
        self.optimizer = optimizer(self.model.parameters(), lr=self.learning_rate)
        self.epochs = epochs

    def setup_model(self):          
        """Creates model template""" 

        layers = []
        for i in range(self.hyper_parameters["n_layers"]):
            layers.append(nn.Linear(self.layers[i], self.layers[i+1]))
            #layers.append(nn.Dropout(p=self.dropout_rate))
            layers.append(nn.ReLU())
        
        layers.append(nn.Linear(self.layers[i+1], self.out_features))
        model = nn.Sequential(*layers)

        return model
    
    def predict(self, X, y):
        """Makes prediction"""
        X = convert_to_tensor(X)
        y_pred = self.model(X)

        _, y_pred = torch.max(y_pred, 1)
        
        return Predictions(data=X, y_true=y, y_pred=y_pred)
    
    def plot_loss_history(self, work_dir):
        # figure dimensions
        plt.figure(figsize=(10, 7))
        # generate plot
        plt.plot(range(1, self.epochs+1), self.loss_history)
        # set labels for axis and title
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
    """
    Function takes as input a pandas dataframe and
    converts it to a pytorch tensor. If the input is 
    the target values then use proper format.
    """
    if target:
        return torch.LongTensor(df["condition"].values)
    else:
        return torch.FloatTensor(df.values)
