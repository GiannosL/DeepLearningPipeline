import os
import numpy as np
from source.setup.terminal_colours import terminal_colors as tc


def create_file_structure(db_path):
    directories_to_create = ("database/", "database/training/", "database/testing/")
    
    for dire in directories_to_create:
        try:
            os.mkdir(f"{db_path}/{dire}")
        except:
            print(tc.warning + f"[X] Directory '{db_path}/{dire}' already exists!" + tc.endc)
            return 0


def open_file(filename):
    with open(filename, "r") as f:
        contents = f.read()
        contents = contents.splitlines()
    
    headers = [i for i in contents[0].split(",")]
    return contents[1:], headers


def read_csv(filename):
    f, headers = open_file(filename)
    columns = [[] for i in headers]

    for line in f:
        new_line = line.split(",")
        for i, val in enumerate(new_line):
            columns[i].append(val)
    
    return headers, columns


def read_tsv(filename):
    f, headers = open_file(filename)
    columns = [[] for i in headers]

    for line in f:
        new_line = line.split("\t")
        for i, val in enumerate(new_line):
            columns[i].append(val)
    
    return headers, columns


def split_data(split_perc: float, indiv_number: int):
    # do an 80-20 split
    train_indivs = int(indiv_number * split_perc)
    print(tc.bold + f"Number of individuas: {indiv_number}")
    print(f"Split: {train_indivs} training, {indiv_number-train_indivs} testing" + tc.endc)

    train_indices = np.random.choice(range(0, indiv_number), size=train_indivs, replace=False)
    test_indices = []
    for i in range(indiv_number):
        if i not in train_indices:
            test_indices.append(i)
    
    return train_indices, test_indices


def save_contents(type_string: str, header, contents_col, indices, db_path: str):
    
    for i, attribute in enumerate(header):
        with open(f"{db_path}/database/{type_string}/feature_{attribute}.csv", "w+") as f:
            for x, j in enumerate(contents_col[i]):
                if x in indices:
                    f.write(f"{j}\n")


def generate_yaml(database_path: str, feature_names: list):
    training_module = "training:\n"
    testing_module = "testing:\n"

    for feature in feature_names:
        training_module += f"  {feature}: '{database_path}/database/training/feature_{feature}.csv'\n"
        testing_module += f"  {feature}: '{database_path}/database/testing/feature_{feature}.csv'\n"
    
    yaml_file = training_module + "\n" + testing_module

    with open(f"{database_path}/database/features.yaml", "w+") as f:
        f.write(yaml_file)


def setup_database(database_path: str, input_file: str, train_perc: float):
    #
    print(tc.okgreen + "[ ] Creating file structure" + tc.endc)
    create_file_structure(db_path = database_path)

    #
    print(tc.okgreen + "[ ] Reading input file" + tc.endc)
    file_ext = input_file.split(".")[-1]
    if file_ext == "csv":
        headers, file_contents = read_csv(input_file)
    elif file_ext == "tsv":
        headers, file_contents = read_tsv(input_file)
    else:
        err = tc.fail + "File does not end with '.tsv' or '.csv'. "
        err += "Please format and rename file accordingly!" + tc.endc
        raise Exception(err)
    
    #
    print(tc.okgreen + "[ ] Splitting data into train and test" + tc.endc)
    train_indices, test_indices = split_data(train_perc, len(file_contents[0]))

    #
    print(tc.okgreen + "Saving results" + tc.endc)
    save_contents("training", headers, file_contents, train_indices, database_path)
    save_contents("testing", headers, file_contents, test_indices, database_path)

    #
    print(tc.okgreen + "Creating yaml file" + tc.endc)
    generate_yaml(database_path, feature_names=headers)
