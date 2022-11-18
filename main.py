#!/usr/bin/env python3
from source.model.network import ANN
from source.setup.setup import setup_run
from source.setup.arguments import collect_arguments
from source.output.report_generation import make_report
from source.setup.terminal_colours import terminal_colors
from source.hpo.hyper_param_opt import Hyper_Parameter_Optimization
from source.data.data_preparation import generate_datasets, standardize_data

# collect input arguments
args = collect_arguments()

# read configuration
config = setup_run(args.configfile) 

# build datasets
training_set, test_set = generate_datasets(continuous_feature_list=config["features_continuous"].split(","), 
                                           categorical_feature_list=config["features_categorical"].split(","),
                                           database_yaml=config["database_yaml"])

# plot principal components of the dataset
print(terminal_colors.okcyan + "[ ] Calculationg PCs of input dataset.\n" + terminal_colors.endc)
training_set.plot_principal_components(save_path=config["work_directory"])
test_set.plot_principal_components(save_path=config["work_directory"], train_test_flag="test")

# perform hyper parameter optimization
hpo = Hyper_Parameter_Optimization(training_set, n_trials=config["hpo_trials"], model_name=config["model_name"])
hpo.data.show_results()

# start up main model
my_model = ANN(input_layer_nodes=training_set.feature_number, hyper_params=hpo.data.param_set,
               output_layer_nodes=training_set.class_number, name=config["model_name"])
my_model.setup_training(learning_rate=hpo.data.param_set["learning_rate"], epochs=hpo.data.param_set["n_epochs"])

# Normalize data
training_set, test_set = standardize_data(training_set, test_set)

# train model, use standardized data
my_model.train_model(X=training_set.feature_matrix, y_true=training_set.target,
                     work_directory=config["work_directory"])

# run on test set
preds = my_model.predict(X=test_set.feature_matrix, y=test_set.target)
preds.plot(plt_name=f"{config['work_directory']}results/plots/discrimination_pcs.png")

# generate report
make_report(config["work_directory"], model=my_model, trn_data=training_set, tst_data=test_set)
