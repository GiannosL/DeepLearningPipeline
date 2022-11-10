import torch
import torch.nn as nn
import torch.nn.functional as F

from model_preparation import convert_to_tensor


def build_mock_model(in_features, out_features, params):
    model = nn.Sequential(
        nn.Linear(in_features, params["n_units"]),
        F.relu(),
        nn.Linear(params["n_units"], out_features),
    )
    return model


def train_model(X, y, model, epochs=50, itta=0.01, 
                criterion=nn.CrossEntropyLoss(), 
                optimizer=torch.optim.Adam):

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
    
    with torch.no_grad():
        y_pred = model.forward(X)
    
    return model, loss_history
