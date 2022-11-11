import os, yaml

def read_configuration(filename):
    with open(filename, "r") as f:
        config = yaml.full_load(f)
    
    return config


def setup_run(config_file):

    # get input files
    config = read_configuration(config_file)

    # set-up file structure
    try:
        os.mkdir(config["work_directory"])
        print(f"\n[ ] Creating \"{config['work_directory']}\"!\n[ ] Proceeding to read input data...")
    except:
        print(f"\n[x] Directory \"{config['work_directory']}\" already exists!")
        print("[ ] Proceeding to read input data...")
    
    return config
