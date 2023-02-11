
#%%
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt 
import tabular_data
import numpy as np

def split_data(X, y):
    '''Splits Test, Train and Validation data'''
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    X_test, X_validation, y_test, y_validation = train_test_split(X_test, y_test, test_size=0.5)
    return X_train, y_train, X_test, y_test, X_validation, y_validation

if __name__ == '__main__':

    # load the data
    file = 'clean_tabular_data.csv'
    df = tabular_data.read_csv_data(file)
    X,y = tabular_data.load_airbnb(df)

    # split data in train, test and validation sets
    X_train, y_train, X_test, y_test, X_validation, y_validation = split_data(X,y)

    # select model
    model = SGDRegressor()

    # fitting the model
    model.fit(X_train, y_train)

    # estimate predicted labels
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    y_validation_pred = model.predict(X_validation)

    # Model Performance: RMSE
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    validation_rmse = np.sqrt(mean_squared_error(y_validation, y_validation_pred)) 
    print(f"Train RMSE: {train_rmse:.2f} | Validation RMSE: {validation_rmse:.2f} | Test RMSE: {test_rmse:.2f}"   )

    # Model Performance: R-squared
    score_train = model.score(X_train, y_train)
    score_test = model.score(X_test, y_test)
    score_validation = model.score(X_validation, y_validation)
    print(f"R-squared; Train: {score_train:.2f} | Validation : {score_validation:.2f} | Test : {score_test:.2f} ")

"""     # plot predicted and original labels for test data
    x_ax = range(len(y_test))
    plt.plot(x_ax, y_test, linewidth=1, label="original 'test'")
    plt.plot(x_ax, y_test_pred, linewidth=1.1, label="predicted 'test'")
    plt.legend(loc='best',fancybox=True, shadow=True)
    plt.title("y-test and y-predicted data")
    plt.xlabel('X-axis')
    plt.ylabel('Price_Night')
    plt.show() """