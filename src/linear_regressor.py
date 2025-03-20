import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict, KFold

from pyawd import VectorAcousticWaveDataset3D



def main():
    """ Handle the main workflow of the script """
    nb_samples = 50  # Number of samples in the dataset
    PLOT_GRAPHS = True  # Set to True to plot graphs
    PLOT_BY_FOLD = False  # Set to True to plot graphs by fold
    X,y, interrogators = init_dataset(nb_samples)

    # Define KFold 
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    # Cross validation 
    regressor  = LinearRegression()
    score = cross_val_score(regressor, X, y, cv=kf, scoring="neg_mean_squared_error")
    print("Cross Validation Score :",score)
    y_pred = cross_val_predict(regressor, X, y, cv=kf)

    # Compute absolute error
    errors = np.mean(np.abs(y_pred - y), axis=1)  # Average error per experiment
    # Compute NMSE per experiment
    #errors = NMSE_per_experiment(y, y_pred)

    best_seismometer = best_interrogator(X, y, interrogators, kf)
    print(f"Best seismometer position based on the MSE comparison: {best_seismometer}")

    if PLOT_GRAPHS:
        if PLOT_BY_FOLD:
            # Assign colors to each fold
            colors = ["blue", "green", "red", "purple", "orange"]
            folds = np.zeros(len(y), dtype=int)  # Store fold index for each sample

            for fold_idx, (_, test_idx) in enumerate(kf.split(X)):
                folds[test_idx] = fold_idx  # Assign fold number to test samples

            plot_linear_reg_by_fold(y, y_pred, colors, folds)
            plot_abs_error_analysis_by_fold(errors, colors, folds)
        else:
            plot_linear_reg(y, y_pred)
            plot_abs_error_analysis(errors)
    

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
        interrogator1 = experiment[1][(10, 0, 0)].T  # Get the interrogator data of the first seismometer
        interrogator2 = experiment[1][(-10, 0, 0)].T  # Get the interrogator data of the second seismometer
        X.append(np.hstack((interrogator1,interrogator2))) # Concatenate the two interrogator data
    X = np.array(X)  # Convert to NumPy
    y = np.array(y)

    #print("y shape :",y.shape)
    #print("X shape :",X.shape)

    # Reshaping data
    X = X.reshape(X.shape[0], -1)  # Flatten to (nb_samples, nb_features)
    return X, y, interrogators



def best_interrogator(X, y, interrogators, kf):
    """ Determine the best interrogator based on the mean squared error """
    # Calculate the mean squared error of each Interrogator
    nb_features_per_interrogator = X.shape[1] // 2  # Number of features per interrogator

    X1 = X[:, :nb_features_per_interrogator]  # Data from Interrogator 1
    X2 = X[:, nb_features_per_interrogator:]  # Data from Interrogator 2

    reg1 = LinearRegression()
    reg2 = LinearRegression()

    mse1 = -np.mean(cross_val_score(reg1, X1, y, cv=kf, scoring="norm_mean_squared_error"))
    mse2 = -np.mean(cross_val_score(reg2, X2, y, cv=kf, scoring="norm_mean_squared_error"))
        
    print("MSE 1 :", mse1)
    print("MSE 2 :", mse2)

    best_interrogator = interrogators[0] if mse1 < mse2 else interrogators[1]
    return best_interrogator


def plot_linear_reg(y, y_pred):
    """ Plot the predicted vs. true values """
    # Old plotting the true values against the predicted values
    plt.figure(figsize=(8, 6))
    plt.scatter(y.flatten(), y_pred.flatten(), color='g', alpha=0.5, label="Predictions")  # Scatter plot
    plt.plot(y.flatten(), y.flatten(), color='k', linestyle="--", label="Perfect fit")  # Diagonal line
    plt.xlabel("True values")
    plt.ylabel("Predicted values")
    plt.title("Cross-validation: Predicted vs True Values")
    plt.legend()
    plt.show()

def plot_abs_error_analysis(errors):
    """ Plot the prediction error per experiment index as a bar chart """
    # Plot error per experiment index as a bar chart
    plt.figure(figsize=(8, 6))
    plt.bar(range(len(errors)), errors, color='r', alpha=0.7, label="Absolute Error")
    plt.xlabel("Experiment Index")
    plt.ylabel("Prediction Error (Absolute)")
    plt.title("Prediction Error per Experiment")
    plt.legend()
    plt.show()

def plot_linear_reg_by_fold(y, y_pred, colors, folds):
    """ Plot the predicted vs. true values, coloring each fold differently """
    # Plot the predicted vs. true values, coloring each fold differently
    plt.figure(figsize=(8, 6))

    for fold_idx in range(5):
        plt.scatter(
            y[folds == fold_idx].flatten(),
            y_pred[folds == fold_idx].flatten(),
            color=colors[fold_idx],
            alpha=0.7,
            label=f"Fold {fold_idx+1}"
        )
    plt.plot(y.flatten(), y.flatten(), color='k', linestyle="--", label="Perfect fit")  
    plt.xlabel("True values")
    plt.ylabel("Predicted values")
    plt.title("Cross-validation: predicted vs true values (colored by fold)")
    plt.legend()
    plt.show()

def plot_abs_error_analysis_by_fold(errors, colors, folds):
    """ Plot the prediction error per experiment index as a bar chart with fold colors """
    # Plot error per experiment index as a bar chart with fold colors
    plt.figure(figsize=(8, 6))

    # Create bars with different colors for each fold
    for fold_idx in range(5):
        plt.bar(
            np.where(folds == fold_idx)[0],  # Select indices for the current fold
            errors[folds == fold_idx],  # Select error values for the fold
            color=colors[fold_idx],  # Assign the same color as the scatter plot
            alpha=0.7,
            label=f"Fold {fold_idx+1}"
        )

    plt.xlabel("Experiment index")
    plt.ylabel("Prediction error (absolute)")
    plt.title("Prediction error per experiment (colored by fold)")
    plt.legend()
    plt.show()


# Compute Normalized MSE
def NMSE(y, y_pred):
    """ Compute the normalized mean squared error """
    mse = mean_squared_error(y, y_pred)
    variance = np.var(y)
    nmse = mse / variance
    return nmse

def NMSE_per_experiment(y, y_pred):
    """ Compute the normalized mean squared error per experiment """
    mse = np.mean((y - y_pred) ** 2, axis=1)  # Compute MSE per experiment
    variance = np.var(y, axis=1)  # Compute variance per experiment
    nmse = mse / variance  # Normalize
    return nmse

if __name__ == "__main__":
    main()