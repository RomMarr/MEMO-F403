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
from utils import *
from test_models import Tests



def main():
    """ Handle the main workflow of the script """
    nb_samples = 256  # Number of samples in the dataset  -> TBD : 1/4 of the total of the whole position possible
    PLOT_GRAPHS = True  # Set to True to plot graphs
    PLOT_LATEX = True  # Set to True to plot LaTeX tabular
    
    X,y, interrogators = init_dataset(nb_samples)

    


    # Define KFold 
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    tests = Tests(X, y, kf)
    errors_for_graph, names_for_graph, errors_for_latex, names_for_latex = tests.get_errors()
    y_pred = tests.y_pred

    if PLOT_GRAPHS :
        plot(y, y_pred, errors_for_graph, names_for_graph)
    if PLOT_LATEX :
        generate_latex_table(errors_for_latex, names_for_latex)
    

def init_dataset(samples):
    """ Initialize the dataset and return the features, target, and interrogators """
    # Load the dataset
    interrogators = [(10, 0, 0), (-10, 0, 0)]
    dataset = VectorAcousticWaveDataset3D(samples, interrogators=interrogators)

    # Reshape position data to 2D (1, 3) and repeat to match the rows of interrogators
    pos1 = np.array(interrogators[0]).reshape(1, 3)  # Shape (1, 3)
    pos2 = np.array(interrogators[1]).reshape(1, 3)  # Shape (1, 3)
    
    # Initialize lists to store features and target
    X = []  # Features
    y = []  # Target responses

    for idx in range(samples):
        y.append(dataset.get_epicenter(idx))
        experiment = dataset[idx]  # Get the experiment data
        interrogator1 = experiment[1][interrogators[0]].T  # Get the interrogator data of the first seismometer
        interrogator2 = experiment[1][interrogators[1]].T  # Get the interrogator data of the second seismometer
        pos1_repeated = np.tile(pos1, (interrogator1.shape[0], 1))  # Repeat for the number of data points in interrogator1
        pos2_repeated = np.tile(pos2, (interrogator2.shape[0], 1))  # Repeat for the number of data points in interrogator2
        
        X.append(np.hstack((interrogator1, interrogator2, pos1_repeated, pos2_repeated)))
    X = np.array(X)  # Convert to NumPy
    y = np.array(y)

    #print("X shape:", X.shape)
    #print("y shape:", y.shape)

    # Reshaping data
    X = X.reshape(X.shape[0], -1)  # Flatten to (nb_samples, nb_features)
    return X, y, interrogators







# def cross_validate_nn(X, y, kf, n_epochs=500):
#     """Perform K-Fold Cross-Validation for the Neural Network"""
#     nmse_scores = []
#     for train_idx, test_idx in kf.split(X):
#         X_train, X_test = X[train_idx], X[test_idx]
#         y_train, y_test = y[train_idx], y[test_idx]

#         # Train the NN and get predictions
#         y_pred_test = train_nn(X_train, y_train, X_test, X.shape[1], n_epochs)

#         # Compute NMSE for this fold
#         nmse = NMSE(y_test, y_pred_test)
#         nmse_scores.append(nmse)
#     return np.mean(nmse_scores)





if __name__ == "__main__":
    main()