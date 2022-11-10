import optuna
from hpo_results import HPO_Study

from model_for_hpo import build_mock_model, train_model


def objective(trial, model=None):
    # set parameters to optimize
    params = {
        "n_units": trial.suggest_int("u_units", 5, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.0001, 0.01)
    }

    # model builder -- mock
    model = build_mock_model(4, 3, params)

    #
    X, y = "", ""
    train_model(X, y, model, itta=params["learning_rate"])
    
    return model


def hyper_parameter_optimization(data, number_of_trials=100):
    """Example of hyper parameter optimization"""
    # initialize optuna study
    study = optuna.create_study()
    
    # begin optimization
    study.optimize(objective, n_trials=number_of_trials)

    #
    results = HPO_Study(study.best_trial.number, study.best_params, 
                        study.best_value, study.trials, "Pythagoras")

    return results


# Run HPO
hyper_parameters = hyper_parameter_optimization(number_of_trials=100)
hyper_parameters.hpo_results()
hyper_parameters.plot_top_trials()
