import os
import argparse
from pathlib import Path
from yaml import safe_load

from source import terminal_colors


class SetUpUtilities:
    @staticmethod
    def collect_arguments():
        """
        argument parser for configuration file
        """
        # create parser
        parser = argparse.ArgumentParser()
        # add an argument
        parser.add_argument("--configfile", type=str, required=True)
        
        return parser.parse_args()
    
    @staticmethod
    def read_configuration(filename: str) -> Path | None:
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
