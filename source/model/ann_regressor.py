import torch
import torch.nn as nn
from pathlib import Path
import matplotlib.pyplot as plt

from source.data.data import Dataset
from source import TerminalColours as tc


class ANN_Regressor:
    def __init__(self, 
                 dataset: Dataset) -> None:
        self._data = dataset
        
        self.model = self._build_model(n_input_features=dataset.input_features(),
                                       n_output_features=1,
                                       n_hidden_features=(dataset.input_features()*2))
        
        # model training parameters
        self._lr = 0.01
        self._epochs = 10000
        self._opt = torch.optim.SGD(self.model.parameters(), lr=self._lr)
        self._loss = nn.MSELoss()
        
        print(f'{tc.okgreen}Model is ready!{tc.endc}')
    
    def _build_model(self, 
                     n_input_features: int,
                     n_output_features: int,
                     n_hidden_features: int) -> nn.Sequential:
        """
        create the Neural Network
        """
        layers: list = [
            nn.Linear(in_features=n_input_features, 
                      out_features=n_hidden_features),
            nn.LeakyReLU(),
            nn.Linear(in_features=n_hidden_features, 
                      out_features=n_hidden_features),
            nn.LeakyReLU(),
            nn.Linear(in_features=n_hidden_features, 
                      out_features=n_output_features),
        ]

        return nn.Sequential(*layers)
    
    def train(self) -> None:
        """
        train model on dataset
        """
        # prepare data
        training_set = self._data.training_set()
        x = torch.from_numpy(training_set['X'].to_numpy()).float()
        y = [[i] for i in training_set['y']]
        y = torch.tensor(y).float()

        # start training
        loss_history: list = []
        for ep in range(self._epochs):
            # predict
            y_pred = self.model(x)
            # evaluate model performance through loss
            loss = self._loss(y_pred, y.float())
            loss_history.append(loss.item())
            # back propagation
            self.model.zero_grad()
            loss.backward()
            self._opt.step()
        
        print(f'{tc.okgreen}Model training is complete!{tc.endc}')
        print(f'Loss in final epoch: {loss_history[-1]}')
        self.plot_loss_history(loss_history=loss_history)
    
    def save_model(self, name: str) -> None:
        torch.save(self.model.state_dict(), f'{name}.pt')
        print(f'{tc.okgreen}Model {name} is saved!{tc.endc}')
    
    def load_model(self, pt_file: Path) -> None:
        self.model.load_state_dict(torch.load(pt_file))

    def plot_loss_history(self, loss_history: list[float]) -> None:
        plt.plot(range(1, self._epochs+1), loss_history)
        plt.title('Loss history')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.show()
