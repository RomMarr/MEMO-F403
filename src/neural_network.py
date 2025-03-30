import torch
import torch.nn as nn
import torch.optim as optim

class NeuralNetwork(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.l1 = nn.Linear(input_size, 10)
        self.l2 = nn.Linear(10, 3)  # Predicting 3D epicenter coordinates
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.tanh(self.l1(x))
        x = self.l2(x)
        return x

        
def train_nn(X_train, y_train, X_test, input_size, n_epochs=500):
    """Train the neural network and return test predictions"""
    model = NeuralNetwork(input_size)
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

    for _ in range(n_epochs):
        optimizer.zero_grad()
        y_pred_train = model(X_train_tensor)
        loss = loss_fn(y_pred_train, y_train_tensor)
        loss.backward()
        optimizer.step()

    # Get predictions for the test set
    y_pred_test = model(X_test_tensor).detach().numpy()
    return y_pred_test




