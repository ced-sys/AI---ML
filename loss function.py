import numpy as np

# 1. Mean Squared Error (MSE) Loss
def mse_loss(y_true, y_pred):
    """
    Calculate Mean Squared Error between true and predicted values
    y_true: array of true values
    y_pred: array of predicted values
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.mean((y_true - y_pred) ** 2)

# Derivative of MSE Loss
def mse_loss_derivative(y_true, y_pred):
    """
    Derivative of MSE with respect to predictions
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return 2 * (y_pred - y_true) / len(y_true)

# 2. Binary Cross-Entropy Loss
def binary_cross_entropy_loss(y_true, y_pred):
    """
    Calculate Binary Cross-Entropy between true labels and predictions
    y_true: array of true binary labels (0 or 1)
    y_pred: array of predicted probabilities (between 0 and 1)
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    # Clip predictions to avoid log(0)
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

# Derivative of Binary Cross-Entropy Loss
def binary_cross_entropy_derivative(y_true, y_pred):
    """
    Derivative of Binary Cross-Entropy with respect to predictions
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
    return (y_pred - y_true) / (y_pred * (1 - y_pred) * len(y_true))

# 3. Categorical Cross-Entropy Loss
def categorical_cross_entropy_loss(y_true, y_pred):
    """
    Calculate Categorical Cross-Entropy for multi-class problems
    y_true: array of one-hot encoded true labels
    y_pred: array of predicted probabilities for each class
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
    return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))

# Derivative of Categorical Cross-Entropy Loss
def categorical_cross_entropy_derivative(y_true, y_pred):
    """
    Derivative of Categorical Cross-Entropy with respect to predictions
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
    return (y_pred - y_true) / len(y_true)

# Example usage
def test_loss_functions():
    # Test data
    y_true_regression = [1, 2, 3]
    y_pred_regression = [1.1, 1.9, 3.2]
    
    y_true_binary = [0, 1, 1]
    y_pred_binary = [0.1, 0.9, 0.8]
    
    y_true_categorical = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    y_pred_categorical = [[0.7, 0.2, 0.1], [0.1, 0.8, 0.1], [0.2, 0.3, 0.5]]

    # Test MSE
    mse = mse_loss(y_true_regression, y_pred_regression)
    mse_grad = mse_loss_derivative(y_true_regression, y_pred_regression)
    print(f"MSE Loss: {mse}")
    print(f"MSE Derivative: {mse_grad}")

    # Test Binary Cross-Entropy
    bce = binary_cross_entropy_loss(y_true_binary, y_pred_binary)
    bce_grad = binary_cross_entropy_derivative(y_true_binary, y_pred_binary)
    print(f"Binary Cross-Entropy Loss: {bce}")
    print(f"BCE Derivative: {bce_grad}")

    # Test Categorical Cross-Entropy
    cce = categorical_cross_entropy_loss(y_true_categorical, y_pred_categorical)
    cce_grad = categorical_cross_entropy_derivative(y_true_categorical, y_pred_categorical)
    print(f"Categorical Cross-Entropy Loss: {cce}")
    print(f"CCE Derivative:\n{cce_grad}")

if __name__ == "__main__":
    test_loss_functions()