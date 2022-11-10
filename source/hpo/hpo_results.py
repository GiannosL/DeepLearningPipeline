import numpy as np
import matplotlib.pyplot as plt


class HPO_Study:
    def __init__(self, best_trial, best_param_set, best_loss, all_trials, model_name):
        self.name = model_name
        self.trial = best_trial
        self.param_set = best_param_set
        self.loss = best_loss
        self.trials = all_trials
    
    def show_results(self):
        print("\n-------------------------------------------------------------------")
        print(f"Hyper Parameter Optimization for {self.name}")
        print(f"Number of trials:\t{len(self.trials)}")
        print(f"Best performing trial:\t{self.trial}")
        print(f"Lowest loss achieved:\t{self.loss}")
        for param in self.param_set:
            print(f"Best value for {param}:\t{self.param_set[param]}")
        print("-------------------------------------------------------------------\n")

    def plot_top_trials(self, n_best_trials=5):
        """Not ready for deployment yet!"""
        all_losses = [self.trials[i].values[0] for i in range(len(self.trials))]

        all_losses = np.array(all_losses)
        top_n_indices = np.argsort(all_losses)[:n_best_trials]

        return 0
