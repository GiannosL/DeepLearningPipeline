import pandas as pd
from pathlib import Path
from pandas import DataFrame
from sklearn.model_selection import train_test_split

from source.setup.configuration import ConfigurationSetup


class Dataset:
    def __init__(self, configuration: ConfigurationSetup) -> None:
        self._df = self._parse_dataset(dp=configuration['database'],
                                       continutous_features=configuration['features_continuous'],
                                       categorical_features=configuration['features_categorical'])
        
        # seperate the target column from the features
        X_total, y_total = self._split_target_features(
            features=[*configuration['features_continuous'], 
                      *configuration['features_categorical']],
            target_col='target'
        )

        #
        self._train, self._test = self._train_validation_test_split(X=X_total, 
                                                                    y=y_total,
                                                                    test_s=0.2, 
                                                                    rand_st=42)

    def _parse_dataset(self, 
                       dp: Path, 
                       continutous_features: list[str], 
                       categorical_features: list[str]) -> DataFrame:
        """
        parse input dataset with proper datatypes
        """
        data_types: dict = {}
        for item in continutous_features:
            data_types[item] = float
        for item in categorical_features:
            data_types[item] = object
        
        return pd.read_csv(dp, dtype=data_types)
    
    def _split_target_features(self,
                               features: list[str],
                               target_col: str) -> tuple:
        """
        split target from features
        """
        # valiate the existence of columns in dataframe
        if target_col not in self._df.columns:
            raise ValueError(f'Column \"{target_col}\" not in dataset.')
        #
        for element in features:
            if element not in self._df.columns:
                raise ValueError(f'Column \"{element}\" not in dataset.')
        return self._df[features], self._df[target_col]
    
    def _train_validation_test_split(self,
                                     X, y, 
                                     test_s: float,
                                     rand_st: int) -> tuple[dict]:
        """
        split into a training-validation set and
        a test set
        """
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=test_s, 
            random_state=rand_st
            )
        
        return {'X': X_train, 'y': y_train}, {'X': X_test, 'y': y_test}
    
    def training_set(self) -> dict[str, DataFrame]:
        return self._train
    
    def test_set(self) -> dict[str, DataFrame]:
        return self._test
