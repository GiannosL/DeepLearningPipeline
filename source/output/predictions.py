import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from source.setup.terminal_colours import terminal_colors


class Predictions:
    def __init__(self, data, y_true, y_pred):

        # check that the true target values have been provided
        if not y_true.empty:
            self.y_true = y_true
            self.calculate_stats(y_true, y_pred)
        else:
            self.y_true = pd.DataFrame()
            self.accuracy = 0

        # save the prediction results
        self.data = data 
        self.y_pred = y_pred
    
    def calculate_stats(self, y_true, y_pred):
        """Calculate useful statistics for model evaluation"""
        results = y_true
        results["prediction"] = y_pred

        results["cond_pred_sub"] = results["condition"] - results["prediction"]

        correct = results[results["cond_pred_sub"] == 0]
        
        self.accuracy = (correct.shape[0]/y_true.shape[0])*100
        
        print(f"{terminal_colors.bold}\n-------------------------------------------------------------------{terminal_colors.endc}")
        print(f"Correct predictions / all: {correct.shape[0]} / {y_pred.shape[0]}")
        print(f"Test prediction accuracy: {self.accuracy}%")
        print(f"{terminal_colors.bold}-------------------------------------------------------------------\n{terminal_colors.endc}")

    def pca(self):
        n_feat = self.data.shape[1]
        pca = PCA(n_components=n_feat)

        pcs = pca.fit_transform(self.data)
        pcs = pd.DataFrame(pcs, columns=[f"PC-{i+1}" for i in range(n_feat)])
        eigvals = pca.explained_variance_ratio_
        return pcs, eigvals
    
    def plot(self, plt_name):
        if self.y_true.empty:
            return 0
        
        fig, ax = plt.subplots(1, 2, figsize=(10, 7))
        pcs, eigvals = self.pca()
        pcs["y_true"] = self.y_true["condition"]
        pcs["y_pred"] = self.y_pred

        ax[0].scatter(pcs["PC-1"], pcs["PC-2"], c=pcs["y_true"], edgecolors="black")
        ax[0].set_ylabel(f"PC-2, {np.round(eigvals[1]*100, 3)}%", fontsize=12)

        ax[1].scatter(pcs["PC-1"], pcs["PC-2"], c=pcs["y_pred"], edgecolors="black")

        fig.suptitle("Prediction ability comparisons", fontsize=16, weight="bold")
        ax[0].set_xlabel(f"PC-1, {np.round(eigvals[0]*100, 3)}%", fontsize=12)
        ax[1].set_xlabel(f"PC-1, {np.round(eigvals[0]*100, 3)}%", fontsize=12)
        
        plt.savefig(plt_name, dpi=1000)
        plt.clf()
