import numpy as np
from source.model.network import ANN

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


def edit_model_file(working_directory, model: ANN):
    model_file = read_html(template_name="source/templates/model.html")
    
    # model description
    model_file = model_file.replace("_B1_", f"{model.in_features} nodes")
    model_file = model_file.replace("_B2_", f"{model.h1} nodes")
    model_file = model_file.replace("_B3_", f"{model.h2} nodes")
    model_file = model_file.replace("_B4_", f"{model.h3} nodes")
    model_file = model_file.replace("_B5_", f"{model.out_features} nodes")
    model_file = model_file.replace("_B6_", f"-")
    model_file = model_file.replace("_B7_", f"{np.round(model.learning_rate, 5)}")
    # training stats
    model_file = model_file.replace("_C1_", f"{model.epochs}")
    model_file = model_file.replace("_C2_", f"{model.training_set_size} samples")
    model_file = model_file.replace("_C3_", f"{np.round(model.training_loss, 5)}")

    # loss over epochs
    model_file = model_file.replace("_TRAINING_CURVE_", f"{working_directory}results/plots/training_loss.png")

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


def make_report(working_directory, model):

    main_file = edit_main_file(working_directory)
    model_file = edit_model_file(working_directory, model)
    results_file = edit_results_file()

    #
    save_html(f"{working_directory}report/main_report.html", main_file)
    save_html(f"{working_directory}report/model_report.html", model_file)
    save_html(f"{working_directory}report/results_report.html", results_file)
