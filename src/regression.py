import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score, cross_val_predict, KFold
from pyawd import VectorAcousticWaveDataset3D
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from plot import *
from neural_network import train_nn



def main():
    """ Handle the main workflow of the script """
    nb_samples = 50  # Number of samples in the dataset  -> TBD : 1/4 of the total of the whole position possible
    PLOT_GRAPHS = True  # Set to True to plot graphs
    X,y, interrogators = init_dataset(nb_samples)
    errors = []
    models_names = ["Linear Regression", "Gradient Boosting", "Random Forest", "Neural Network"]
    regressors = [LinearRegression(), MultiOutputRegressor(GradientBoostingRegressor(random_state=42)),RandomForestRegressor(random_state=42)]

    # Define KFold 
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    # Cross validation 
    for regressor in regressors:
        print(regressor)
        score = cross_val_score(regressor, X, y, cv=kf, scoring="neg_mean_squared_error")
        print("Cross Validation Score :",score)
        y_pred = cross_val_predict(regressor, X, y, cv=kf)
        nmse_value = NMSE(y, y_pred)
        errors.append(nmse_value)
    print("Errors : ", errors)

    nmse_nn = cross_validate_nn(X, y, kf, n_epochs=500)
    print("Neural Network NMSE:", nmse_nn)
    errors.append(nmse_nn)



    if PLOT_GRAPHS :
        plot(X, y, y_pred, errors, models_names)
    

def init_dataset(samples):
    """ Initialize the dataset and return the features, target, and interrogators """
    # Load the dataset
    samples = 50  
    interrogators = [(10, 0, 0), (-10, 0, 0)]
    dataset = VectorAcousticWaveDataset3D(samples, interrogators=interrogators)
    # Initialize lists to store features and target
    X = []  # Features
    y = []  # Target responses
    for idx in range(samples):
        y.append(dataset.get_epicenter(idx))
        experiment = dataset[idx]  # Get the experiment data
        interrogator1 = experiment[1][interrogators[0]].T  # Get the interrogator data of the first seismometer
        interrogator2 = experiment[1][interrogators[1]].T  # Get the interrogator data of the second seismometer
        X.append(np.hstack((interrogator1,interrogator2))) # Concatenate the two interrogator data
    X = np.array(X)  # Convert to NumPy
    y = np.array(y)

    #print("y shape :",y.shape)
    #print("X shape :",X.shape)

    # Reshaping data
    X = X.reshape(X.shape[0], -1)  # Flatten to (nb_samples, nb_features)
    return X, y, interrogators




# Compute Normalized MSE
def NMSE(y, y_pred):
    """ Compute the normalized mean squared error """
    mse = mean_squared_error(y, y_pred)
    variance = np.var(y)
    nmse = mse / variance
    return nmse

def cross_validate_nn(X, y, kf, n_epochs=500):
    """Perform K-Fold Cross-Validation for the Neural Network"""
    nmse_scores = []
    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Train the NN and get predictions
        y_pred_test = train_nn(X_train, y_train, X_test, X.shape[1], n_epochs)

        # Compute NMSE for this fold
        nmse = NMSE(y_test, y_pred_test)
        nmse_scores.append(nmse)
    return np.mean(nmse_scores)

def plot(X,y, y_pred, errors, models_names):
    # corrected way to plot scatter graph :

    y_x, y_y, y_z = y[:,0], y[:,1], y[:,2]
    y_pred_x, y_pred_y, y_pred_z = y_pred[:,0], y_pred[:,1], y_pred[:,2]
    plot_linear_reg(y_x, y_pred_x, " (x-coordonates)")
    plot_linear_reg(y_y, y_pred_y, " (y-coordonates)")
    plot_linear_reg(y_z, y_pred_z, " (z-coordonates)")
    
    plot_NMSE_error_analysis(errors, models_names)




if __name__ == "__main__":
    main()