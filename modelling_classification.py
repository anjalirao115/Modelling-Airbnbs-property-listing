#%%
import os
import json
import joblib
import sklearn
import modelling
import numpy as np
import tabular_data
import hyperparameter
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

def make_predictions(model, X_train, X_test, X_validation):
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    y_validation_pred = model.predict(X_validation)
    return y_train_pred, y_test_pred, y_validation_pred

def get_metrics_for_classification_model(y_train, y_train_pred, y_test, y_test_pred, y_validation, y_validation_pred):
    performance_metric = {}
    set_names = ['train', 'test', 'validation']
    sets = [(y_train, y_train_pred), (y_test, y_test_pred), (y_validation, y_validation_pred)]

    for i in range(len(sets)):
        y, y_hat = sets[i]
        
        accuracy    = accuracy_score(y, y_hat)
        precision   = precision_score(y, y_hat, average="macro")
        recall      = recall_score(y, y_hat, average="macro")
        f1          = f1_score(y, y_hat, average="macro")

        performance_metric[f"accuracy_{set_names[i]}"]  = accuracy
        performance_metric[f"precision_{set_names[i]}"] = precision
        performance_metric[f"recall_{set_names[i]}"]    = recall
        performance_metric[f"f1_{set_names[i]}"]        = f1

    return performance_metric

def tune_classification_model_hyperparameters(model, X_train, y_train, X_test, y_test, X_validation, y_validation, parameter_grid:dict):
    grid_search = sklearn.model_selection.GridSearchCV(
        estimator = model,
        param_grid = parameter_grid
    )

    grid_search = grid_search.fit(X_train, y_train)

    # finding the best model and hyperparameters from grid_search
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_

    y_train_pred, y_test_pred, y_validation_pred = make_predictions(best_model, X_train, X_test, X_validation)
    performance_metric = get_metrics_for_classification_model(y_train, y_train_pred, y_test, y_test_pred, y_validation, y_validation_pred)

    return best_model, best_params, performance_metric

def evaluate_all_cls_models(model_list, parameter_grid_list):

    # looping over models and corresponding parameter grids
    for i in range(len(model_list)):

        model = model_list[i]               
        parameter_grid = parameter_grid_list[i]

        # finds the best model, parameters and performance metrics
        best_model, best_params, performance_metric = tune_classification_model_hyperparameters(model, X_train, y_train, X_test, y_test, X_validation, y_validation, parameter_grid)
        
        # defines folder name to save the model, parameter and metrics
        model_name = str(model)[:-2]        
        folder_path = f"models/classification/{model_name}"

        os.makedirs(folder_path, exist_ok=True)

        # Saves the model, parameter and metrics in a folder
        hyperparameter.save_model(folder_path, best_model, best_params, performance_metric)
        print(f"Best model, parameters and metrics saved for {model}")
        print("--" * 10)

def find_best_cls_model(model_list, folder):
    dict = {} # dictionary for model name and validation_accuracy pairs

    for model in model_list:
        model_name = str(model)[:-2]
        metric_files = f"{folder}/{model_name}/metrics.json"

        with open(metric_files) as json_file:
            metric = json.load(json_file)

        dict[model_name] = metric['accuracy_validation']

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
    
    # select model
    model = LogisticRegression(max_iter=200)

    # fitting the model
    model.fit(X_train, y_train)

    # estimate predicted labels
    y_train_pred, y_test_pred, y_validation_pred = make_predictions(model, X_train, X_test, X_validation)
    
    # get performance metrics for baseline classification model
    performance_metric = get_metrics_for_classification_model(y_train, y_train_pred, y_test, y_test_pred, y_validation, y_validation_pred)

    # printing the metrics
    print(f"\t Training \t Validation \t Test")
    print(f" Accuracy \t {performance_metric['accuracy_train']:.2f} \t {performance_metric['accuracy_test']:.2f}  \t {performance_metric['accuracy_validation']:.2f}")
    print(f" Precision \t {performance_metric['precision_train']:.2f} \t {performance_metric['precision_test']:.2f}  \t {performance_metric['precision_validation']:.2f}")
    print(f" Recall \t {performance_metric['recall_train']:.2f} \t {performance_metric['recall_test']:.2f}  \t {performance_metric['recall_validation']:.2f}")
    print(f" F1  \t\t {performance_metric['f1_train']:.2f} \t {performance_metric['f1_test']:.2f}  \t {performance_metric['f1_validation']:.2f}")

    parameter_grid={"C":np.logspace(-3,3,7), "penalty":["l2"]}

    best_model, best_params, performance_metric = tune_classification_model_hyperparameters(model, X_train, y_train, X_test, y_test, X_validation, y_validation, parameter_grid)
    folder_path = "models/classification/logistic_regression"

    os.makedirs(folder_path, exist_ok=True)

    # Saves the tuned model, parameter and metrics in a folder
    hyperparameter.save_model(folder_path, best_model, best_params, performance_metric)

    #Now tuning three other models: decision trees, random forests, and gradient boosting

    # DecisionTreeClassifier
    parameter_grid_dtc={
        "criterion": ['gini', 'entropy'],
        "max_depth": range(1,10),
        "min_samples_split": range(1,10),
        "min_samples_leaf": range(1,5)
    }
    
    # RandomForestClassifier
    parameter_grid_rfc = { 
        'n_estimators': [200, 500],
        'max_depth' : [4,5,6],
        'criterion' :['gini', 'entropy']
    }

    #GradientBoostingClassifier
    parameter_grid_gbc = {
        "n_estimators": [25, 50],
        "loss": ["log_loss"]
    }

    model_list = [
        DecisionTreeClassifier(),
        RandomForestClassifier(),
        GradientBoostingClassifier()
    ]

    parameter_grid_list= [parameter_grid_dtc, parameter_grid_rfc, parameter_grid_gbc]

    evaluate_all_cls_models(model_list, parameter_grid_list)
    
    folder = "models/classification"
    best_cls_model, parameters, performance_metric = find_best_cls_model(model_list, folder)

    print(f"The best classification model is {best_cls_model}")
    print(f"Performance metric for the best model is {performance_metric}")



# %%
