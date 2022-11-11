import torch
import pandas as pd
import torch.nn as nn
import matplotlib.pyplot as plt
from source.predictions import Predictions


def convert_to_tensor(df: pd.DataFrame, target=False):
    if target:
        return torch.LongTensor(df["condition"].values)
    else:
        return torch.FloatTensor(df.values)


def train_model(X, y, model, epochs=50, itta=0.01, 
                criterion=nn.CrossEntropyLoss(), 
                optimizer=torch.optim.Adam,
                plot_loss=False, plot_name="results/plots/training_loss.png"):

    X = convert_to_tensor(X)
    y = convert_to_tensor(y, target=True)
    
    loss_history = []
    optimizer = optimizer(model.parameters(), lr=itta)

    for i in range(epochs):
        i += 1
        y_pred = model.forward(X)

        # 
        loss = criterion(y_pred, y)
        loss_history.append(loss.item())

        # backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    if plot_loss:
        plot_loss_history(epochs, loss_history, plot_name)
    
    return model, loss_history


def make_prediction(X, model, y: pd.DataFrame() = pd.DataFrame()):
    X = convert_to_tensor(X)

    with torch.no_grad():
        y_pred = model.forward(X)
    
    #y_pred = pd.DataFrame(y_pred.numpy())
    _, y_pred = torch.max(y_pred, 1)
    
    return Predictions(data=X, y_true=y, y_pred=y_pred)


def plot_loss_history(epochs, losses, plt_name, save_fn=True):
    plt.figure(figsize=(10, 7))
    plt.ylim((0, 1))
    plt.plot(range(1, epochs+1), losses)
    plt.title("Loss trajectory over epochs", fontsize=16, weight="bold")
    plt.ylabel("Loss", fontsize=12)
    plt.xlabel("Epochs", fontsize=12)
    if save_fn:
        plt.savefig(plt_name, dpi=1200)
        plt.clf()
    else:
        plt.show()
