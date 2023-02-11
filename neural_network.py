#%%
import os
import time
import yaml
import json
import itertools
import numpy as np
import modelling
import tabular_data
from datetime import datetime
import torch
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import torch.nn.functional as F


# Dataloader class
class AirbnbNightlyPriceDataset(Dataset):
    def __init__(self, feature, label):     #input type is pandas dataframe: X and y
        super().__init__()
        self.X = torch.tensor(feature.values, dtype=torch.float32)
        self.y = torch.tensor(label.values, dtype=torch.float32)

    def __getitem__(self, index):
        feature = self.X[index]
        label = self.y[index]
        return (feature, label)

    def __len__(self):
        return len(self.X)

# Model class
class Feedforward(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        # Define model layers
        width = config["hidden_layer_width"]
        depth = config["depth"]
        layers = []
        layers.append(torch.nn.Linear(11, width))
        for hidden_layer in range(depth - 1):
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Linear(width, width))
        layers.append(torch.nn.ReLU())
        layers.append(torch.nn.Linear(width, 1))
        self.layers = torch.nn.Sequential(*layers)

    def forward(self, X):
        # Using the layers to process the features
        output = self.layers(X)
        return output

def train(model, dataloader, config:dict, epochs:int):
    """ The function trains the model and adds loss to tensorboard"""
    optimiser = torch.optim.SGD(model.parameters(), lr=config["learning_rate"])
    writer = SummaryWriter()    # to write losses to tensorboard

    for epoch in range(epochs):
        for batch_idx, batch in enumerate(dataloader):
            features, labels = batch            # tensors: features:[16,11], labels:[16]
            features = features.type(torch.float32)
            # Giving labels the same shape as predictions/features
            labels = torch.unsqueeze(labels, 1) # labels tensor size now is [16, 1]

            # Forward pass
            prediction = model(features)
            loss = F.mse_loss(prediction, labels.float())
            # Backward pass
            loss.backward()
            
            # Gradient Optimisation 
            optimiser.step() 
            optimiser.zero_grad()
            # Adds loss to tensorboard
            writer.add_scalar("loss", loss.item(), batch_idx)
            

def get_nn_config(config_yaml_file):
    """ Reads the model configuration dict from yaml file """
    with open(config_yaml_file, "r") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
        return config         

def get_rmse_r2_score(model, X, y):
    X = torch.tensor(X.values).type(torch.float32)
    y = torch.tensor(y.values).type(torch.float32)
    y = torch.unsqueeze(y, 1)
    y_hat = model(X)
    rmse_loss = torch.sqrt(F.mse_loss(y_hat, y.float()))
    r2_score = 1 - rmse_loss / torch.var(y.float())
    return rmse_loss, r2_score

def get_performance_metric(model, training_duration, epochs, X_train, y_train, X_val, y_val, X_test, y_test):
    # Define the dictionary, and later just add key, value pairs
    metrics_dict = {"training_duration_s": training_duration}
    
    number_of_predictions = epochs * len(X_train)
    inference_latency = training_duration / number_of_predictions
    
    train_rmse, train_r2_score  = get_rmse_r2_score(model, X_train, y_train)
    val_rmse,   val_r2_score    = get_rmse_r2_score(model, X_val, y_val)
    test_rmse,  test_r2_score   = get_rmse_r2_score(model, X_test, y_test)

    print(f"Train RMSE: {train_rmse.item():.2f} | Train R2: {train_r2_score.item():.2f}")
    print(f"Validation RMSE: {val_rmse.item():.2f} | Validation R2: {val_r2_score.item():.2f}")
    print(f"Test RMSE: {test_rmse.item():.2f} | Test R2: {test_r2_score.item():.2f}")

    # Adds key, value pairs to the metrics dictionary
    metrics_dict["inference_latency"] = inference_latency

    metrics_dict["train_rmse"]      = train_rmse.item()
    metrics_dict["validation_rmse"] = val_rmse.item()
    metrics_dict["test_rmse"]       = test_rmse.item()

    metrics_dict["train_r2_score"]      = train_r2_score.item()
    metrics_dict["validation_r2_score"] = val_r2_score.item()
    metrics_dict["test_r2_score"]       = test_r2_score.item()

    return metrics_dict     

def save_model(model, hyperparam_dict, metrics_dict, folder):
    """ Saves the model, param_dict and metrics_dict in target folder."""
    if not isinstance(model, torch.nn.Module):
        print("Error: Model is not a Pytorch module")
    else:
        # Target directory is named using current time
        time_date = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
        folder = f"{folder}/{time_date}"
        # Creates the directory
        os.makedirs(folder, exist_ok=True)
        # Saves model
        torch.save(model.state_dict(), f"{folder}/model.pt")
        # Saves hyperparameters
        with open(f"{folder}/hyperparameters.json", 'w') as fp:
            json.dump(hyperparam_dict, fp)
        # Saves performance metrics
        with open(f"{folder}/metrics.json", 'w') as fp:
            json.dump(metrics_dict, fp)

def generate_nn_configs():
    ''' Finds all possible combinations of hyperparameters
    and returns a list of dictionaries '''
    param_space = {
    'optimizer': ['Adam', 'AdamW'],
    'learning_rate': [0.001, 0.01],
    'hidden_layer_width': [10, 20],
    'depth': [4, 6]
    }
    # Finds all combindations of hyperparameters    
    keys, values = zip(*param_space.items())
    param_dict_list = [dict(zip(keys, v)) for v in itertools.product(*values)]
    return param_dict_list

if __name__ == '__main__':

    file = 'clean_tabular_data.csv'
    df = tabular_data.read_csv_data(file) 
    X,y = tabular_data.load_airbnb(df)      

    # data sets: type is pandas dataframe
    X_train, y_train, X_test, y_test, X_val, y_val = modelling.split_data(X,y)

    # Define Datasets
    dataset_train   = AirbnbNightlyPriceDataset(X_train,y_train)
    dataset_test    = AirbnbNightlyPriceDataset(X_test,y_test)
    dataset_val     = AirbnbNightlyPriceDataset(X_val, y_val)

    # Define the dataloaders
    batch_size = 16
    dataloader_train    = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    dataloader_test     = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)
    dataloader_val      = DataLoader(dataset_val, batch_size=batch_size, shuffle=False)       

    config  = get_nn_config("nn_config.yaml")
    model   = Feedforward(config)
    epochs  = 200

    model_folder = "models/regression/neural_networks"

    param_dict_list = generate_nn_configs()

    lowest_RMSE_loss_validation = np.inf #set the bar very low for losses and reach the smaller values of loss in loop
    for idx, param_dict in enumerate(param_dict_list):
        print(f"Training loop for param configuration {idx+1}/{len(param_dict_list)}")
        model = Feedforward(param_dict)
        start_time = time.time()
        train(model, dataloader_train, param_dict, epochs)
        end_time = time.time()
        training_duration = end_time - start_time
        
        metrics_dict = get_performance_metric(model, training_duration, epochs, X_train, y_train, X_val, y_val, X_test, y_test)
        save_model(model, param_dict, metrics_dict, model_folder)
        
        # the parameter for model evaluation is RMSE for validation set, therefore
        val_rmse = metrics_dict["validation_rmse"]
        
        if val_rmse < lowest_RMSE_loss_validation:
            lowest_RMSE_loss_validation = val_rmse
        
            bestmodel_folder = "models/regression/neural_networks/best_model"
            os.makedirs(bestmodel_folder, exist_ok=True)

            if os.path.exists(f"{bestmodel_folder}/model.pt") == False:
                print("Best model written")
            else:
                print("Best model overwritten")

            # Saves best model
            torch.save(model.state_dict(), f"{bestmodel_folder}/model.pt")
            # Saves hyperparameters
            with open(f"{bestmodel_folder}/hyperparameters.json", 'w') as fp:
                json.dump(param_dict, fp)
            # Saves performance metrics
            with open(f"{bestmodel_folder}/metrics.json", 'w') as fp:
                json.dump(metrics_dict, fp)
        print("--" * 10)
