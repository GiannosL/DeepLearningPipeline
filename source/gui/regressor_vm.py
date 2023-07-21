"""
Regressor view model
"""
from source.data.data import Dataset
from source.model.ann_regressor import ANN_Regressor

class RegressorViewModel:
    def __init__(self, data: Dataset) -> None:
        # create model
        model = ANN_Regressor(dataset=data)

        # train model
        model.train()
