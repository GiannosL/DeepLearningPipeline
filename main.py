#!/usr/bin/env python3
from source.model.network import ANN
from source.setup.setup import setup_run
from source.setup.arguments import collect_arguments
from source.output.report_generation import make_report
from source.setup.terminal_colours import terminal_colors
from source.hpo.hyper_param_opt import Hyper_Parameter_Optimization
from source.data.data_preparation import generate_datasets, standardize_data

from source.DeepLearner import DeepLearner

# collect input arguments
args = collect_arguments()

# read configuration
config = setup_run(args.configfile) 

# set-up my model
study_model = DeepLearner(configuration=config)

# build dataset
study_model.prepare_input()

# perform hyper parameter optimization
study_model.optimize_hyper_parameters()

# train model
study_model.train_model()

# run on test set
study_model.make_predictions()

# save prediction results and generate report
study_model.save_results()
