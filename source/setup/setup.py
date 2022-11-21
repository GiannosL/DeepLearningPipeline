import os, yaml
from source.setup.terminal_colours import terminal_colors

def read_configuration(filename):
    with open(filename, "r") as f:
        config = yaml.full_load(f)
    
    if not config["work_directory"].endswith("/"):
        config["work_directory"] += "/"
    
    return config


def create_directories(work_dir):
    try:
        os.mkdir(work_dir)
        print(f"{terminal_colors.okgreen}[ ] Creating \"{work_dir}\"!{terminal_colors.endc}")
    except:
        print(f"{terminal_colors.warning}[x] Directory \"{work_dir}\" already exists!{terminal_colors.endc}")


def setup_run(config_file):

    # get input files
    config = read_configuration(config_file)

    if "model_name" not in config.keys():
        config["name"] = "Pythagoras"

    # set-up file structure
    print("\n")
    create_directories(config["work_directory"])
    create_directories(f"{config['work_directory']}report/")
    create_directories(f"{config['work_directory']}results/")
    create_directories(f"{config['work_directory']}results/plots/")
    print("\n")
    
    return config
