import numpy as np
from source.model.network import ANN
from source.data.features import Input_data
from source.data.data_preparation import barplot_pc_variance


def read_html(template_name):
    with open(template_name, "r") as f:
        f = f.read()
    return f


def save_html(save_file, contents):
    with open(save_file, "w+") as f:
        f.write(contents)


def create_html_table_rows(pc_dict_trn, pc_dict_tst):
    table_str = ""
    for k in pc_dict_trn.keys():
        table_str += f"<tr><td>{k}</td><td>{pc_dict_trn[k]}</td><td>{pc_dict_tst[k]}</td></tr>"
    
    return table_str


def edit_main_file(working_directory, trn_data: Input_data, tst_data: Input_data):
    #
    training_pc_var_dict = trn_data.pc_variance()
    test_pc_var_dict = tst_data.pc_variance()
    #
    main_file = read_html(template_name="source/templates/main.html")

    #
    main_file = main_file.replace("_PC_VALUES_", create_html_table_rows(training_pc_var_dict, test_pc_var_dict))
    barplot_pc_variance(working_directory, training_pc_var_dict, test_pc_var_dict)

    #
    main_file = main_file.replace("_TRAIN_PC1_PC2_PATH_", f"{working_directory}results/plots/training_data_pcs.png")
    main_file = main_file.replace("_TEST_PC1_PC2_PATH_", f"{working_directory}results/plots/test_data_pcs.png")
    main_file = main_file.replace("_PC_VARIANCE_", f"{working_directory}results/plots/pc_variance.png")
    
    main_file = main_file.replace("main.html", "main_report.html")
    main_file = main_file.replace("model.html", "model_report.html")
    main_file = main_file.replace("results.html", "results_report.html")
    return main_file


def edit_model_file(working_directory, model: ANN):
    model_file = read_html(template_name="source/templates/model.html")
    
    # model description
    model_file = model_file.replace("_MODEL_NAME_", model.name)
    model_file = model_file.replace("_B1_", f"{model.in_features} nodes")
    model_file = model_file.replace("_B2_", f"{model.h1} nodes")
    model_file = model_file.replace("_B3_", f"{model.h2} nodes")
    model_file = model_file.replace("_B4_", f"{model.h3} nodes")
    model_file = model_file.replace("_B5_", f"{model.out_features} nodes")
    model_file = model_file.replace("_B6_", f"{model.dropout_rate}")
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


def edit_results_file(working_directory):
    results_file = read_html(template_name="source/templates/results.html")

    results_file = results_file.replace("_CLASSIFICATION_RESULTS_PC_PLOT_", f"{working_directory}results/plots/discrimination_pcs.png")

    results_file = results_file.replace("main.html", "main_report.html")
    results_file = results_file.replace("model.html", "model_report.html")
    results_file = results_file.replace("results.html", "results_report.html")
    return results_file


def make_report(working_directory, model, trn_data: Input_data, tst_data: Input_data):

    main_file = edit_main_file(working_directory, trn_data, tst_data)
    model_file = edit_model_file(working_directory, model)
    results_file = edit_results_file(working_directory)

    #
    save_html(f"{working_directory}report/main_report.html", main_file)
    save_html(f"{working_directory}report/model_report.html", model_file)
    save_html(f"{working_directory}report/results_report.html", results_file)
