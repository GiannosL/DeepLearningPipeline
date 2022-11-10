import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

yaml_file = "dataset/features.yaml"
with open(yaml_file, "r") as f:
    data_dictionary = yaml.full_load(f)

class Feature:
    def __init__(self, name, dataset_flag="train", target_feature=False):
        self.name = name
        self.target_feature = target_feature
        if dataset_flag=="train":
            self.filename = data_dictionary["training"][name]
        else:
            self.filename = data_dictionary["testing"][name]

        self.data = self.read_csv()
    
    def read_csv(self):
        data = pd.read_csv(self.filename, header=None)
        data = data.rename(columns={0:self.name})
        
        self.sample_number = data.shape[0]
        return data


class Input_data:
    def __init__(self):
        self.feature_number = 0
        self.sample_number = 0

        self.column_names = []
        self.target_flag = False

        self.feature_matrix = pd.DataFrame()

        self.class_number = 0
        self.target = None

    def add_feature(self, new_feature: Feature):
        if self.feature_number:
            self.sample_number = new_feature.sample_number
        else:
            assert(new_feature.sample_number != self.sample_number)
        
        if new_feature.target_feature:
            self.target = new_feature.data
            self.target_flag = True
            self.get_class_number()
        
        else:
            self.feature_matrix[new_feature.name] = new_feature.data
            self.feature_number += 1
            self.column_names.append(new_feature.name)
    
    def get_class_number(self):
        self.class_number = self.target["condition"].nunique()

    def standardize(self):
        """This will be used exclusively for the PCA"""
        standardizer = StandardScaler()
        std_data = standardizer.fit_transform(self.feature_matrix)
        std_data = pd.DataFrame(std_data, columns=self.column_names)

        return std_data
    
    def pca(self, normalize):
        pca_model = PCA(n_components=self.feature_number)

        data = self.feature_matrix
        if normalize == True:
            data = self.standardize()

        principal_components = pca_model.fit_transform(data)
        principal_components = pd.DataFrame(data=principal_components,
                                        columns=[f"PC-{i}" for i in range(1, self.feature_number+1)])
        eigenvalues = pca_model.explained_variance_ratio_

        return principal_components, eigenvalues

    def plot_principal_components(self, save_path, train_test_flag="training", a=1, b=2, normalize=True):
        plt.figure(figsize=(10, 7)) 
        a = f"PC-{a}"
        b = f"PC-{b}"

        # target_list
        target_list = self.target["condition"].tolist()

        # PCA
        pcs, eigvals = self.pca(normalize)

        if self.target_flag:
            pc1_null, pc1_alt = [], []
            pc2_null, pc2_alt = [], []

            pc1 = pcs[a].to_list()
            pc2 = pcs[b].to_list()
            
            for i, value in enumerate(target_list):
                if value == 0:
                    pc1_null.append(pc1[i])
                    pc2_null.append(pc2[i])
                elif value == 1:
                    pc1_alt.append(pc1[i])
                    pc2_alt.append(pc2[i])

            plt.scatter(pc1_null, pc2_null, c="orange", edgecolors="black")
            plt.scatter(pc1_alt, pc2_alt, c="green", edgecolors="black")
        
        else:
            plt.scatter(pcs[a], pcs[b], c="grey", edgecolors="black")
        
        if normalize:
            plt.title(f"{train_test_flag.capitalize()} data - Standardized", weight="bold", fontsize=16)
        else:
            plt.title(f"{train_test_flag.capitalize()} data - Not standardized", weight="bold", fontsize=16)
        
        plt.xlabel(f"{a}, {np.round(eigvals[0]*100, 2)}", fontsize=12)
        plt.ylabel(f"{b}, {np.round(eigvals[1]*100, 2)}", fontsize=12)
        plt.savefig(f"{save_path}{train_test_flag}_data_pcs.png", dpi=1200)
        plt.clf()
