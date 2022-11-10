def read_html(template_name):
    with open(template_name, "r") as f:
        f = f.read()
    return f


def save_html(save_file, contents):
    with open(save_file, "w+") as f:
        f.write(contents)


def edit_main_file(working_directory):
    main_file = read_html(template_name="source/templates/main.html")
    main_file = main_file.replace("_TRAIN_PC1_PC2_PATH_", f"{working_directory}training_data_pcs.png")
    main_file = main_file.replace("_TEST_PC1_PC2_PATH_", f"{working_directory}test_data_pcs.png")
    return main_file


def make_report(working_directory):

    main_file = edit_main_file(working_directory)

    #
    save_html(f"{working_directory}main_report.html", main_file)
