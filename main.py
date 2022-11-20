#!/usr/bin/env python3
from source.setup.setup import setup_run
from source.setup.arguments import collect_arguments


from source.DeepLearner import DeepLearner

# collect input arguments
args = collect_arguments()

# read configuration
config = setup_run(args.configfile) 

# set-up my model
study_model = DeepLearner(configuration=config)

"""
# do auto run
study_model.run_auto()
"""

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
