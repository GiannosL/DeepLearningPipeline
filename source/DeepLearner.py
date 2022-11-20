from source.model.network import ANN
from source.output.report_generation import make_report
from source.setup.terminal_colours import terminal_colors
from source.hpo.hyper_param_opt import Hyper_Parameter_Optimization
from source.data.data_preparation import generate_datasets, standardize_data

class DeepLearner:
    def __init__(self, configuration: dict):
        self.config = configuration
        self.model_name = configuration["model_name"]
        self.work_directory = configuration["work_directory"]
    
    def prepare_input(self):
        self.training_set, self.test_set = generate_datasets(continuous_feature_list=self.config["features_continuous"].split(","), 
                                           categorical_feature_list=self.config["features_categorical"].split(","),
                                           database_yaml=self.config["database_yaml"])
        
        # make principal component plot of the input
        self.visualize_input()
    
    def visualize_input(self):
        print(terminal_colors.okcyan + "[ ] Calculationg PCs of input dataset.\n" + terminal_colors.endc)
        self.training_set.plot_principal_components(save_path=self.work_directory)
        self.test_set.plot_principal_components(save_path=self.work_directory, train_test_flag="test")
    
    def optimize_hyper_parameters(self):
        # perform hyper parameter optimization
        hpo = Hyper_Parameter_Optimization(self.training_set, n_trials=self.config["hpo_trials"], model_name=self.model_name)
        hpo.data.show_results()
        hpo.data.plot_trials(working_directory=self.work_directory)
        # save the results of HPO within the object
        self.hpo = hpo
    
    def train_model(self):
        """
        Initialize model, set-up hyper-parameters based 
        on HPO and finall train the model
        """
        my_model = ANN(input_layer_nodes=self.training_set.feature_number, hyper_params=self.hpo.data.param_set,
               output_layer_nodes=self.training_set.class_number, name=self.model_name)
        my_model.setup_training(cv_split=self.hpo.cross_validation_split, learning_rate=self.hpo.data.param_set["learning_rate"], 
                        epochs=self.hpo.data.param_set["n_epochs"])
        
        # Normalize data
        self.training_set, self.test_set = standardize_data(self.training_set, self.test_set)

        # train model, use standardized data
        my_model.train_model(X=self.training_set.feature_matrix, y_true=self.training_set.target,
                            work_directory=self.config["work_directory"])
        
        self.model = my_model
    
    def make_predictions(self):
        # run on test set
        preds = self.model.predict(X=self.test_set.feature_matrix, y=self.test_set.target)
        preds.plot(plt_name=f"{self.work_directory}results/plots/discrimination_pcs.png")

        self.predictions = preds
    
    def save_results(self):
        # save prediction results
        self.predictions.save_results(working_dir=self.work_directory, model_name=self.model_name)

        # save model
        self.model.save(f"{self.work_directory}results/{self.model_name}")

        # generate report
        make_report(self.work_directory, model=self.model, trn_data=self.training_set, tst_data=self.test_set)
