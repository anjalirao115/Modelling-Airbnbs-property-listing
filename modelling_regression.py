#%%
import os
import json
import joblib
import modelling
import tabular_data
import hyperparameter
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor

def evaluate_all_models(model_list, parameter_grid_list):

    # looping over models and corresponding parameter grids
    for i in range(len(model_list)):

        model = model_list[i]               
        parameter_grid = parameter_grid_list[i]

        # finds the best model, parameters and performance metrics
        best_model, best_params, performance_metric = hyperparameter.tune_regression_model_hyperparameters(model, X_train, y_train, X_test, y_test, X_validation, y_validation, parameter_grid)
        
        # defines folder name to save the model, parameter and metrics
        model_name = str(model)[:-2]        
        folder_path = f"models/regression/{model_name}"

        os.makedirs(folder_path, exist_ok=True)

        # Saves the model, parameter and metrics in a folder
        hyperparameter.save_model(folder_path, best_model, best_params, performance_metric)
        print(f"Best model, parameters and metrics saved for {model}")
        print("--" * 10)

def find_best_model(model_list, folder):
    dict = {} # dictionary for model name and validation_accuracy pairs

    for model in model_list:
        model_name = str(model)[:-2]
        metric_files = f"{folder}/{model_name}/metrics.json"

        with open(metric_files) as json_file:
            metric = json.load(json_file)

        dict[model_name] = metric['validation_RMSE']

    best_model_name = min(dict, key=dict.get)

    with open(f"{folder}/{best_model_name}/hyperparameters.json") as json_file:
            parameters = json.load(json_file)

    with open(f"{folder}/{best_model_name}/metrics.json") as json_file:
            performance_metric = json.load(json_file)

    best_reg_model = joblib.load(f"{folder}/{best_model_name}/model.joblib")
    return best_reg_model, parameters, performance_metric

if __name__ == "__main__":   

    # load the data
    file = 'clean_tabular_data.csv'
    df = tabular_data.read_csv_data(file)
    X,y = tabular_data.load_airbnb(df)
    X_train, y_train, X_test, y_test, X_validation, y_validation = modelling.split_data(X,y)
    
    # DecisionTreeRegressor
    parameter_grid_dtr={"splitter":["best","random"],
                "max_depth" : [1,3],
            "min_samples_leaf":[1,2,3],
            "min_weight_fraction_leaf":[0.1,0.2],  
            "max_features": [1.0],
            "max_leaf_nodes":[None,10,20,30] }

    # RandomForestRegressor
    parameter_grid_rfr={'bootstrap': [True, False],
    'max_depth': [10, None],
    'max_features': [1.0],
    'min_samples_leaf': [1, 2],
    'min_samples_split': [2, 5],
    'n_estimators': [400, 600]}

    # GradientBoostingRegressor
    parameter_grid_gbr={'n_estimators':[500,1000],
        'learning_rate':[.001,0.01],
        'max_depth':[1,2,4],
        'subsample':[.5,.75,1],
        'random_state':[1]}

    model_list = [
        DecisionTreeRegressor(),
        RandomForestRegressor(),
        GradientBoostingRegressor()
    ]

    parameter_grid_list= [parameter_grid_dtr, parameter_grid_rfr, parameter_grid_gbr]

    evaluate_all_models(model_list, parameter_grid_list)

    folder = "models/regression"
    best_reg_model, parameters, performance_metric = find_best_model(model_list, folder)

    print(f"The best regression model is {best_reg_model}")
    print(f"Performance metric for the best model is {performance_metric}")