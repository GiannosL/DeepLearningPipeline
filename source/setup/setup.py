import os, yaml

def read_configuration(filename):
    with open(filename, "r") as f:
        config = yaml.full_load(f)
    
    return config


def create_directories(work_dir):
    try:
        os.mkdir(work_dir)
        print(f"\n[ ] Creating \"{work_dir}\"!\n[ ] Proceeding to read input data...")
    except:
        print(f"\n[x] Directory \"{work_dir}\" already exists!")
        print("[ ] Proceeding to read input data...")


def setup_run(config_file):

    # get input files
    config = read_configuration(config_file)

    # set-up file structure
    create_directories(config["work_directory"])
    create_directories(f"{config['work_directory']}/report/")
    create_directories(f"{config['work_directory']}/results/")
    create_directories(f"{config['work_directory']}/results/plots/")
    
    return config
