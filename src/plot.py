import matplotlib.pyplot as plt

def plot(y, y_pred, errors, models_names):
    # corrected way to plot scatter graph :
    y_x, y_y, y_z = y[:,0], y[:,1], y[:,2]
    y_pred_x, y_pred_y, y_pred_z = y_pred[:,0], y_pred[:,1], y_pred[:,2]
    plot_linear_reg(y_x, y_pred_x, " (x-coordonates)")
    plot_linear_reg(y_y, y_pred_y, " (y-coordonates)")
    plot_linear_reg(y_z, y_pred_z, " (z-coordonates)")
    
    plot_NMSE_error_analysis(errors, models_names)


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



def plot_NMSE_error_analysis(errors, models):
    """ Plot the prediction error per model as a bar chart """
    #print("ERRORS = ", errors)
    #print("NAMES = ", models)
    #errors =  [np.float64(10.541431547645516), np.float64(0.9791755029914951), np.float64(1.0025572102779032), np.float64(1.026377505687507),np.float64(1.133495769516704),np.float64(0.891970710354459),np.float64(0.8709576668143852), np.float64(0.8654524411612651),np.float64(0.8840376445194649)]
   # models = ['LinReg (int=T)','Ridge (α=0.001, int=T)','Lasso (α=0.001, int=T)','KNN (bt, k=3, p=1, dist)','DT (depth=5, feat=sqrt, leaf=4, split=10)','SVM (C=10, eps=0.2)','GB (n=200, lr=0.1, d=3)','RF (n=100, d=10, split=2, leaf=1)', 'ConvMixer (dim=32, d=4, k=3, p=7, lr=1e-4, ep=150)']
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


def generate_latex_table(errors, model_names):
    """ Generate a LaTeX table for a given matrix of errors. """
    print("\\begin{table}[h]")
    print("\\centering")
    print("\\begin{tabular}{|l|c|c|c|c|}")
    print("\\hline")
    print("Model & NMSE & NMSE$_x$ & NMSE$_y$ & NMSE$_z$ \\\\")
    print("\\hline")

    # Add rows with error values
    for i, model in enumerate(model_names):
        nmse_values = " & ".join(f"{val:.4f}" for val in errors[i])
        row_str = f"  {model} & {nmse_values} \\\\"  # Initialize row_str correctly
        print(row_str)  # Print each row

    # End the table
    print("\\hline")
    print("\\end{tabular}")
    print("\\caption{Normalized Mean Squared Errors (NMSE) for Different Models}")
    print("\\label{tab:nmse_results}")
    print("\\end{table}")
