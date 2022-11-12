import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

from source.data.features import Feature, Input_data


def generate_datasets(feature_list):
    training_set = Input_data()
    test_set = Input_data()

    for feat in feature_list:
        training_set.add_feature(Feature(feat, dataset_flag="train"))
        test_set.add_feature(Feature(feat, dataset_flag="test"))

    # add target 
    target = Feature("condition", dataset_flag="train", target_feature=True)
    training_set.add_feature(target)
    target = Feature("condition", dataset_flag="test", target_feature=True)
    test_set.add_feature(target)

    return training_set, test_set


def standardize_data(train: Input_data, test: Input_data):
    std_model = StandardScaler()
    train_tmp = std_model.fit_transform(train.feature_matrix)
    test_tmp = std_model.transform(test.feature_matrix)

    train.feature_matrix = pd.DataFrame(train_tmp, columns=train.column_names)
    test.feature_matrix = pd.DataFrame(test_tmp, columns=test.column_names)


    return train, test

def barplot_pc_variance(work_dir, trn_dict, tst_dict):
    trn_list = list(trn_dict.values())
    tst_list = list(tst_dict.values())

    X = np.arange(len(trn_list))

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.bar(X+0.00, trn_list, color="black", width=0.25, label="training-set", edgecolor="black")
    ax.bar(X+0.25, tst_list, color="orange", width=0.25, label="testing-set", edgecolor="black")
    plt.xticks([i+0.125 for i in range(len(trn_list))], list(trn_dict.keys()))
    plt.title("Variance percentage covered by the PCs", fontsize=16, weight="bold")
    plt.xlabel("Principal components", fontsize=12)
    plt.ylabel("Variance (%)", fontsize=12)
    plt.legend()
    plt.ylim(0, 1)
    plt.savefig(f"{work_dir}results/plots/pc_variance.png", dpi=1200)    