import torch
import optuna
import torch.nn as nn
from sklearn.model_selection import KFold

from source.hpo.hpo_results import HPO_Study
from source.model.network import convert_to_tensor


class Hyper_Parameter_Optimization:
    def __init__(self, data, n_trials, model_name, cv_split=5):
        self.name = model_name

        self.number_of_features = data.feature_number
        self.number_of_classes = data.class_number

        self.cross_validation_split = cv_split

        self.X = convert_to_tensor(data.feature_matrix)
        self.y = convert_to_tensor(data.target, target=True)

        self.data = self.hyper_parameter_optimization(number_of_trials=n_trials)

    def create_model(self, in_features, out_features, params):
        
        layers = []
        for i in range(params["n_layers"]):
            i += 1
            n_units = params[f"n_units_{i}"]
            layers.append(nn.Linear(in_features, n_units))
            layers.append(nn.Dropout(p= params["dropout_rate"]))
            layers.append(nn.ReLU())
            in_features = n_units
        layers.append(nn.Linear(in_features, out_features))

        return nn.Sequential(*layers)

    def train_model(self, model, epochs=50, itta=0.01, 
                    criterion=nn.CrossEntropyLoss(), 
                    optimizer=torch.optim.Adam):
        
        self.hpo_trial_num += 1
        self.loss_history[f"trial_{self.hpo_trial_num}"] = []
        optimizer = optimizer(model.parameters(), lr=itta)

        # Start 5-fold cross validation
        kf = KFold(n_splits=self.cross_validation_split)
        for train_index, test_index in kf.split(self.X):
            X_train, X_test = self.X[train_index], self.X[test_index]
            y_train, y_test = self.y[train_index], self.y[test_index]

            self.loss_history[f"trial_{self.hpo_trial_num}"].append([])

            for i in range(epochs):
                i += 1
                y_pred = model.forward(X_train)

                # 
                loss = criterion(y_pred, y_train)
                self.loss_history[f"trial_{self.hpo_trial_num}"][-1].append(loss.item())

                # backpropagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            with torch.no_grad():
                y_pred = model.forward(X_test)
        
        return criterion(y_pred, y_test)

    def objective(self, trial):
        # set parameters to optimize
        params = {
            "n_epochs": trial.suggest_int("n_epochs", 100, 2000, step=20),
            "n_layers": trial.suggest_int("n_layers", 1, 3),
            "n_units_1": trial.suggest_int("n_units_1", 3, 10),
            "n_units_2": trial.suggest_int("n_units_2", 3, 10),
            "n_units_3": trial.suggest_int("n_units_3", 3, 10),
            "dropout_rate": trial.suggest_float("dropout_rate", 0.0001, 0.5),
            "learning_rate": trial.suggest_float("learning_rate", 0.0001, 0.01)
        }

        # model builder -- mock
        model = self.create_model(self.number_of_features, self.number_of_classes, params)

        #
        accuracy = self.train_model(model, epochs=params["n_epochs"], itta=params["learning_rate"])
        
        return accuracy

    def hyper_parameter_optimization(self, number_of_trials=100):
        """Example of hyper parameter optimization"""
        self.hpo_trial_num = 0
        self.loss_history = {}

        # initialize optuna study
        study = optuna.create_study()
        
        # begin optimization
        study.optimize(self.objective, n_trials=number_of_trials)

        #
        results = HPO_Study(study.best_trial.number, study.best_params, 
                            study.best_value, study.trials, self.name, self.loss_history)

        return results
