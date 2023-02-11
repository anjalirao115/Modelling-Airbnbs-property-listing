# Modelling-Airbnbs-property-listing

The project aims to predict the prices of Airbnb's property listings using machine learning. The three types of models trained for the project are:

1. Regression model
2. Classification model
3. Neural network


## Milestone 1 : Data preparation 
### (code: tabular_data.py)

The raw tabular data provided is stored in tabular_data/listing.csv. The data file was read as a pandas dataframe and Exploratory Data Analysis was performed. There are rows with missing rating values, which were dropped. The code 'tabular_data.py' performs this task and writes cleaned processed data in a file 'clean_tabular_data.csv'.

## Milestone 2 : Processing the images 
### (code: prepare_image_data.py)

The provided data also contains a directory of listing's images. These images are of different size. The images were resized to give a fixed height while maintaining their aspect ratio. The code 'prepare_image_data.py' does this task and processed images are stored in 'tabular_data/processed_images'. The image data is not used in model training.

## Milestone 3 : Creating a regression model 
### (code: modelling.py, hyperparameter.py, modelling_regression.py)

1. Using SGDRegressor without hyperparameter tuning (code: modelling.py) : The baseline model 

The first attempt on developing a framework for regression model was made using the sklearn model SGDRegressor. The code written for this step is available as 'modelling.py'. The model does not perform well returning poor metrics for training data. Clearly model is not able to explain the data well and tuning of hyperparameters is needed.

2. Using SGDRegressor with hyperparameter tuning (code: hyperparameter.py)

A gridsearch was performed on a parameter space. Model with the best combination of hyperparameters was used. The performance metrics slightly improve, however the model still does not explain the data well. The best model, performance metrics and hyperparameters are stored in 'models/regression/SGDregressor'.

3. Using decision trees, random forests, and gradient boosting (code: modelling_regression.py)

Attempts are made with three different regression models viz. DecisionTreeRegressor, RandomForestRegressor and GradientBoostingRegressor. The parameters were tuned for all the models and models, performance metrics and tune parameters were saved in models/regression under respective directory names. The model with the least RMSE for validation set is selected as the best regression model.

## Milestone 4 : Creating a classification model 
### (code: modelling_classification.py)

The four models were trained for this task:
1. LogisticRegression
2. DecisionTreeClassifier
3. RandomForestClassifier
4. GradientBoostingClassifier

The hyperparameters are tuned and models are saved in direcotry 'models/classification'.

## Milestone 5 : Creating a neural network 
### (code: neural_network.py)
Following steps are taken for developing, training and testing the neural network:

### Creating a dataloader class

A dataloader class is defined for converting the features and labels from pandas dataframe to torch tensors.

### Creating a model class

A model class Feedforward is created which defines the layers of the network. The model is defined to have Linear layers and also ReLU as activation layers. The hidden layer width and depth of the neural network are provided through a yaml file.

### Generating a dictionaries of neural network configuraion
There can be different combinations of hyperparameters and therefore all configurations are generated using a function 'generate_nn_config'. The network is trained for each of these configurations in loop.


### Model training
The network is trained for each of the coinfigurations mentioned above. Training is done for a number of epochs after dividing the data into batches, and the losses are measured at each step. These losses are written to tensorboard for visual inspection of network training. The following information is stored in directories named after date-time of training :
1. the best model
2. configuration: hyperparameters
3. performance metrics

### Finding the best network
While the training continues for each configuration in loop, the code evaulates the model by lowest RMSE loss for validation set and copies the best model, hyperparameters and performance metrics in a directory named 'models/regression/neural_networks/best_model'.

If a configuration performs better than the previously saved best model, it is replaced by the new found best model and its metrics.

Once the loop is completed, the best trained neural network is stored in the directory 'models/regression/neural_networks/best_model'.


## Summary
The preliminary results show that the neural network performs better than other models, however it is not conclusive yet. The work is in progress and the codes are being refined.
