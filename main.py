#!/usr/bin/env python3
from source.data.data import Dataset
from source.model.ann_classifier import ANN_Classifier
from source.setup.configuration import ConfigurationSetup

# read configuration
config = ConfigurationSetup()

# dataset 
data = Dataset(configuration=config)

# model building
model = ANN_Classifier(dataset=data)

# train model
model.train()

"""
# do auto run
study_model.run_auto()

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
"""