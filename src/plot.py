import matplotlib.pyplot as plt
import numpy as np

 # Plotting the true values against the predicted values
def plot_linear_reg(y, y_pred, title):#, x_label, y_label):
    plt.figure(figsize=(8, 6))
    plt.scatter(y, y_pred, color='g', alpha=0.5, label="Predictions")  # Scatter plot
    plt.plot(y, y, color='k', linestyle="--", label="Perfect fit")  # Diagonal line
    plt.xlabel("True values" + title)
    plt.ylabel("Predicted values" + title)
    plt.title("Cross-validation: predicted vs true values" + title)
    plt.legend()
    plt.show()



# Bar chart of the NMSE of each model
def plot_NMSE_error_analysis(errors, models):
    """ Plot the prediction error per model as a bar chart """
    print("ERRORS : ", errors)
    plt.figure(figsize=(8, 6))
    
    # Set x-axis labels to model names if provided, otherwise use indices
    plt.bar(models, errors, color='r', alpha=0.7, label="NMSE")
    plt.xlabel("Models")
    plt.ylabel("Normalized mean squared error (NMSE)")
    plt.title("NMSE per model")
    plt.xticks(rotation=45)  # Rotate labels for better readability if needed
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.show()


