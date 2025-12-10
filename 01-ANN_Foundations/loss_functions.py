import numpy as np

def mae(y_true, y_pred):
    """Mean Absolute Error: Good if you have outliers."""
    return np.mean(np.abs(y_true - y_pred))

def mse(y_true, y_pred):
    """Mean Squared Error: Standard choice, punishes large errors heavily."""
    return np.mean(np.square(y_true - y_pred))

def huber_loss(y_true, y_pred, delta=1.0):
    """Huber Loss: Best of both worlds (MSE near 0, MAE far away)."""
    error = y_true - y_pred
    is_small_error = np.abs(error) <= delta
    squared_loss = 0.5 * np.square(error)
    linear_loss = delta * (np.abs(error) - 0.5 * delta)
    
    return np.mean(np.where(is_small_error, squared_loss, linear_loss))

def binary_crossentropy(y_true, y_pred):
    """Used for Yes/No problems (e.g., Customer Churn)."""
    y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def categorical_crossentropy(y_true, y_pred):
    """Used for Multi-Class (e.g., MNIST Digits 0-9)."""
    y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
    return -np.sum(y_true * np.log(y_pred)) / y_true.shape[0]

def hinge_loss(y_true, y_pred):
    """Used for SVMs. Max(0, 1 - y_true * y_pred)."""
    return np.mean(np.maximum(0, 1 - y_true * y_pred))

if __name__ == "__main__":
    y_real = np.array([1.0, 0.0, 1.0])
    y_guess = np.array([0.9, 0.1, 0.4]) # Model made a mistake on the last one

    print(f"MAE Loss: {mae(y_real, y_guess):.4f}")
    print(f"MSE Loss: {mse(y_real, y_guess):.4f}")
    print(f"Huber Loss: {huber_loss(y_real, y_guess):.4f}")
    print(f"Binary Crossentropy: {binary_crossentropy(y_real, y_guess):.4f}")