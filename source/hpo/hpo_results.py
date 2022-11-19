import pandas as pd
import matplotlib.pyplot as plt


class HPO_Study:
    def __init__(self, best_trial, best_param_set, 
                best_loss, all_trials, model_name, 
                max_epochs, loss_history=None):
        self.name = model_name
        self.trial = best_trial
        self.param_set = best_param_set
        self.loss = best_loss
        self.trials = all_trials
        
        self.max_epochs = max_epochs
        self.loss_history = loss_history
    
    def show_results(self):
        print("\n-------------------------------------------------------------------")
        print(f"Hyper Parameter Optimization for {self.name}")
        print(f"Number of trials:\t{len(self.trials)}")
        print(f"Best performing trial:\t{self.trial}")
        print(f"Lowest loss achieved:\t{self.loss}")
        for param in self.param_set:
            print(f"Best value for {param}:\t{self.param_set[param]}")
        print("-------------------------------------------------------------------\n")
    
    def save_trials(self):
        loss_dataframe = pd.DataFrame()
        for key in self.loss_history:
            print(key)
            # Only make use of the first CV-dataset
            loss_dataframe[key] = pd.Series(self.loss_history[key][0])
        
        return loss_dataframe

    def plot_trials(self, working_directory: str):
        """Not ready for deployment yet!"""
        df = self.save_trials()
        
        plt.figure(figsize=(10, 7))
        for i in range(df.shape[1]):
            plt.plot([i for i in range(1, 1+df[f"trial_{i+1}"].shape[0])], df[f"trial_{i+1}"], label=f"trial_{i+1}")
        
        plt.title("Loss trajectory of different trials during HPO", weight="bold", fontsize=16)
        plt.xlabel("Number of epochs", fontsize=12)
        plt.ylabel("Loss", fontsize=12)
        
        plt.ylim((0, 1))
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.tight_layout()
        plt.savefig(f"{working_directory}results/plots/hpo_loss.png", dpi=1200)

        return 0
