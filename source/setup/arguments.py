import argparse

def collect_arguments():
    # create parser
    parser = argparse.ArgumentParser()
    # add an argument
    parser.add_argument("--configfile", type=str, required=True)
    
    return parser.parse_args()
