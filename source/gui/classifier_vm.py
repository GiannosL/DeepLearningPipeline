"""
Classifier view model
"""
from source.data.data import Dataset
from source.model.ann_classifier import ANN_Classifier

class ClassifierViewModel:
    def __init__(self, data: Dataset) -> None:
        # create model
        model = ANN_Classifier(dataset=data)

        # train model
        model.train()
