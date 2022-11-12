import pandas as pd
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
