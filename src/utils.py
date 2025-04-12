import numpy as np
from sklearn.metrics import mean_squared_error


# Compute Normalized MSE
def NMSE(y, y_pred):
    """ Compute the normalized mean squared error """
    mse = mean_squared_error(y, y_pred)
    variance = np.var(y)
    nmse = mse / variance
    return nmse

def all_NMSE(y, y_pred):
    nmse = NMSE(y, y_pred)
    nmse_x = NMSE(y[:, 0], y_pred[:, 0])
    nmse_y = NMSE(y[:, 1], y_pred[:, 1])
    nmse_z = NMSE(y[:, 2], y_pred[:, 2])
    return nmse, nmse_x, nmse_y, nmse_z




    
def get_error_from_name(name,errors,models_names):
    idx = models_names.index(name)
    return errors[idx], name

    

def find_best_model_for_each_metric(errors, model_names):
    """ """
    # Convert errors list to a numpy array for easier manipulation
    errors = np.array(errors)

    # Initialize dictionary to store the best model for each metric
    best_models = {
        'NMSE': None,
        'NMSE_x': None,
        'NMSE_y': None,
        'NMSE_z': None
    }

    best_models_with_name = {
        'NMSE': None,
        'NMSE_x': None,
        'NMSE_y': None,
        'NMSE_z': None
    }

    # Iterate through each column (NMSE, NMSE_x, NMSE_y, NMSE_z)
    for i, metric in enumerate(['NMSE', 'NMSE_x', 'NMSE_y', 'NMSE_z']):
        # Find the index of the minimum value for this metric
        min_idx = np.argmin(errors[:, i])

        # Store the best model and its corresponding error for each metric
        best_models[metric] = errors[min_idx, i]
        best_models_with_name[metric] = model_names[min_idx]

    return best_models, best_models_with_name