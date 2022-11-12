from source.model.network import ANN
import source.model.model_preparation as mp
from source.setup.setup import setup_run
from source.report_generation import make_report
from source.hpo.hyper_param_opt import Hyper_Parameter_Optimization
from source.data_preparation import generate_datasets, standardize_data


# read configuration
config = setup_run("model_configuration.yaml") 

# build datasets
training_set, test_set = generate_datasets(feature_list=config["features"].split(","))

# plot principal components of the dataset
print("\n[ ] Calculationg PCs of input dataset.\n")
training_set.plot_principal_components(save_path=config["work_directory"])
test_set.plot_principal_components(save_path=config["work_directory"], train_test_flag="test")

# perform hyper parameter optimization
hpo = Hyper_Parameter_Optimization(training_set, n_trials=config["hpo_trials"])
hpo.data.show_results()

# start up main model
model = ANN(in_features=training_set.feature_number, 
            h1=hpo.data.param_set["n_units_1"], 
            h2=hpo.data.param_set["n_units_2"], 
            h3=hpo.data.param_set["n_units_3"], 
            out_features=training_set.class_number, 
            learning_rate=hpo.data.param_set["learning_rate"],
            epochs=200)

# Normalize data
training_set, test_set = standardize_data(training_set, test_set)

# train model, use standardized data
trained_model, loss_history = mp.train_model(X=training_set.feature_matrix, y=training_set.target, 
                                             model=model, epochs=model.epochs, plot_loss=True, 
                                             itta=model.learning_rate,
                                             plot_name=f"{config['work_directory']}results/plots/training_loss.png")

# run on test set
preds = mp.make_prediction(X=test_set.feature_matrix, model=trained_model, y=test_set.target)
preds.plot(plt_name=f"{config['work_directory']}results/plots/discrimination_pcs.png")

# generate report
make_report(config["work_directory"], trained_model)
