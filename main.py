import os
import yaml
from source.network import ANN
import source.model_preparation as mp
from source.data_preparation import generate_datasets, standardize_data
from source.report_generation import make_report

# read configuration
with open("model_configuration.yaml", "r") as f:
    config = yaml.full_load(f)

# create output
try:
    os.mkdir(config["work_directory"])
    print(f"\n[ ] Creating \"{config['work_directory']}\"!\n[ ] Proceeding to read input data...")
except:
    print(f"\n[x] Directory \"{config['work_directory']}\" already exists!\n[ ] Proceeding to read input data...")

# build datasets
training_set, test_set = generate_datasets(feature_list=config["features"].split(","))

# plot principal components of the dataset
print("\n[ ] Calculationg PCs of input dataset.\n")
training_set.plot_principal_components(save_path=config["work_directory"])
test_set.plot_principal_components(save_path=config["work_directory"], train_test_flag="test")

# start up main model
model = ANN(in_features=training_set.feature_number, out_features=training_set.class_number)
# give model description
model.model_description()

# Normalize data
training_set, test_set = standardize_data(training_set, test_set)

# train model, use standardized data
trained_model, loss_history = mp.train_model(X=training_set.feature_matrix, y=training_set.target, 
                                             model=model, epochs=200, plot_loss=True,
                                             plot_name=f"{config['work_directory']}training_loss.png")

# run on test set
preds = mp.make_prediction(X=test_set.feature_matrix, model=trained_model, y=test_set.target)
preds.plot(plt_name=f"{config['work_directory']}discrimination_pcs.png")

# generate report
make_report(config["work_directory"])
