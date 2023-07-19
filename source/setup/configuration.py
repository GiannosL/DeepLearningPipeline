import argparse
from pathlib import Path
from yaml import safe_load

from source import NECESSARY_COLUMNS


class ConfigurationSetup:
    def __init__(self) -> None:
        # path to the configuration file
        config_str = self._collect_arguments()
        # configuration as dictionary
        self._config = self._read_configuration(filename=config_str)

    def _collect_arguments(self) -> str:
        """
        argument parser for configuration file
        """
        # create parser
        parser = argparse.ArgumentParser()
        # add an argument
        parser.add_argument("--configfile", type=str, required=True)
        
        return parser.parse_args()['configfile']
    
    def _read_configuration(self, filename: str) -> dict[str,str]:
        """
        parse configuration file and validate filepaths
        """
        # convert to Path object
        my_path = Path(filename)

        # 
        if my_path.exists():
            with open(my_path, 'r') as stream:
                return safe_load(stream)
        
        raise FileNotFoundError(f'Configuration file not found in path: {my_path}')
    
    def _variable_existence(self, config_var_list: list[str]) -> None:
        for element in NECESSARY_COLUMNS:
            if element not in config_var_list:
                raise ValueError(f'Value \"{element}\" not in configuration file.')
    
    def _validate_configuration(self, config: dict) -> dict:
        """
        validate the configuration file and format data
        """
        # validate variable existance
        self._variable_existence(config_var_list=config.keys())

        # build formatted dictionary
        formatted_dict: dict = {}

        for key, value in config.items():
            if key == 'database':
                value = Path(value)
                # validate the path's existence
                if not value.exists():
                    raise FileExistsError(f'Database file not in path: {value}')
            if key in ['features_continuous', 'features_categorical']:
                if value:
                    value = value.split(',')
                else:
                    value = []

            # save key with formatted value
            formatted_dict[key] = value
        
        return formatted_dict
    
    def __getitem__(self, key: str) -> list[str] | Path:
        return self._config[key]
