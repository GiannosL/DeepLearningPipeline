import torch
import optuna
import torch.nn as nn
import torch.nn.functional as F

from source.hpo.hpo_results import HPO_Study
from source.model.model_preparation import convert_to_tensor


class Hyper_Parameter_Optimization:
    def __init__(self, data, n_trials):
        self.X = convert_to_tensor(data.feature_matrix)
        self.y = convert_to_tensor(data.target, target=True)

        self.data = self.hyper_parameter_optimization(number_of_trials=n_trials)

    def build_mock_model(self, in_features, out_features, params):
        model = nn.Sequential(
            nn.Linear(in_features, params["n_units"]),
            nn.LeakyReLU(),
            nn.Linear(params["n_units"], out_features),
        )
        return model

    def train_model(self, model, epochs=50, itta=0.01, 
                    criterion=nn.CrossEntropyLoss(), 
                    optimizer=torch.optim.Adam):
        
        loss_history = []
        optimizer = optimizer(model.parameters(), lr=itta)

        for i in range(epochs):
            i += 1
            y_pred = model.forward(self.X)

            # 
            loss = criterion(y_pred, self.y)
            loss_history.append(loss.item())

            # backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        with torch.no_grad():
            y_pred = model.forward(self.X)
        
        return criterion(y_pred, self.y)

    def objective(self, trial):
        # set parameters to optimize
        params = {
            "n_units": trial.suggest_int("n_units", 5, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.0001, 0.01)
        }

        # model builder -- mock
        model = self.build_mock_model(4, 3, params)

        #
        accuracy = self.train_model(model, itta=params["learning_rate"])
        
        return accuracy

    def hyper_parameter_optimization(self, number_of_trials=100):
        """Example of hyper parameter optimization"""
        # initialize optuna study
        study = optuna.create_study()
        
        # begin optimization
        study.optimize(self.objective, n_trials=number_of_trials)

        #
        results = HPO_Study(study.best_trial.number, study.best_params, 
                            study.best_value, study.trials, "Pythagoras")

        return results
