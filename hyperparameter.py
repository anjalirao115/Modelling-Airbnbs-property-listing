#%%
import os
import json
import joblib
import sklearn
import tabular_data
import modelling 
import numpy as np
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


#def custom_tune_regression_model_hyperparameters(model, X_train, y_train, X_test, y_test, X_validation, y_validation, parameter_grid:dict):
#    pass

def tune_regression_model_hyperparameters(model, X_train, y_train, X_test, y_test, X_validation, y_validation, parameter_grid:dict):
    
    grid_search = sklearn.model_selection.GridSearchCV(
        estimator = model,
        param_grid = parameter_grid
    )

    grid_search = grid_search.fit(X_train, y_train)

    # finding the best model and hyperparameters from grid_search
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    
    y_validation_pred = best_model.predict(X_validation)

    validation_rmse = round(np.sqrt(mean_squared_error(y_validation, y_validation_pred)), 2)
    r2_score_validation = best_model.score(X_validation, y_validation)
    performance_metric = {'validation_RMSE': validation_rmse}
    performance_metric["r2_score_validation"] = r2_score_validation

    return best_model, best_params, performance_metric


def save_model(folder:str, best_model, best_params:dict, performance_metric:dict):

    joblib.dump(best_model, f"{folder}/model.joblib")
    
    with open(f"{folder}/hyperparameters.json", 'w') as fp:
        json.dump(best_params, fp)

    with open(f"{folder}/metrics.json", 'w') as fp:
        json.dump(performance_metric, fp)   


if __name__ == "__main__":

    # load the data
    file = 'clean_tabular_data.csv'
    df = tabular_data.read_csv_data(file)
    X,y = tabular_data.load_airbnb(df)
    X_train, y_train, X_test, y_test, X_validation, y_validation = modelling.split_data(X,y)
    
    # select model
    model = SGDRegressor()

    # A grid of model parameters
    parameter_grid = {
        'power_t': [0.1, 0.2, 0.3, 0.4],
        'eta0': [0.001, 0.01, 0.1],
        'n_iter_no_change': [2, 5, 8, 10, 15],
        'alpha': [0.00001, 0.0001, 0.001, 0.01]
        }

    best_model, best_params, performance_metric = tune_regression_model_hyperparameters(SGDRegressor(), X_train, y_train, X_test, y_test, X_validation, y_validation, parameter_grid)

    print(best_params, performance_metric)

    folder = "models/regression/SGDregressor"
    os.makedirs(folder, exist_ok=True)

    save_model(folder, best_model, best_params, performance_metric)