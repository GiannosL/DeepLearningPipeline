def read_html(template_name):
    with open(template_name, "r") as f:
        f = f.read()
    return f


def save_html(save_file, contents):
    with open(save_file, "w+") as f:
        f.write(contents)


def edit_main_file(working_directory):
    main_file = read_html(template_name="source/templates/main.html")
    main_file = main_file.replace("_TRAIN_PC1_PC2_PATH_", f"{working_directory}results/plots/training_data_pcs.png")
    main_file = main_file.replace("_TEST_PC1_PC2_PATH_", f"{working_directory}results/plots/test_data_pcs.png")
    
    main_file = main_file.replace("main.html", "main_report.html")
    main_file = main_file.replace("model.html", "model_report.html")
    main_file = main_file.replace("results.html", "results_report.html")
    return main_file


def edit_model_file():
    model_file = read_html(template_name="source/templates/model.html")

    model_file = model_file.replace("main.html", "main_report.html")
    model_file = model_file.replace("model.html", "model_report.html")
    model_file = model_file.replace("results.html", "results_report.html")
    return model_file


def edit_results_file():
    results_file = read_html(template_name="source/templates/results.html")

    results_file = results_file.replace("main.html", "main_report.html")
    results_file = results_file.replace("model.html", "model_report.html")
    results_file = results_file.replace("results.html", "results_report.html")
    return results_file


def make_report(working_directory):

    main_file = edit_main_file(working_directory)
    model_file = edit_model_file()
    results_file = edit_results_file()

    #
    save_html(f"{working_directory}report/main_report.html", main_file)
    save_html(f"{working_directory}report/model_report.html", model_file)
    save_html(f"{working_directory}report/results_report.html", results_file)
