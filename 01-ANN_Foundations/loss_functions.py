import numpy as np

# --- 1. The Helper Function for train.py ---
def compute_cost(A2, Y):
    """
    Adapter function to match the signature expected by train.py.
    Arguments:
    A2 -- The predicted output (y_pred), shape (1, number of examples)
    Y -- The true labels (y_true), shape (1, number of examples)
    """
    # Simply call the robust binary_crossentropy function below
    # Note: We swap arguments because binary_crossentropy expects (true, pred)
    return binary_crossentropy(Y, A2)


# --- 2. Advanced Loss Functions (Your Collection) ---

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
    """
    Used for Yes/No problems (e.g., Customer Churn).
    Formula: - (y * log(p) + (1-y) * log(1-p))
    """
    m = y_true.shape[1] if len(y_true.shape) > 1 else y_true.shape[0]
    
    # Clip predictions to prevent Log(0) which gives NaN
    y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
    
    # Calculate loss
    loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    
    return np.squeeze(loss) # Ensures result is a scalar (e.g. 17, not [[17]])

def categorical_crossentropy(y_true, y_pred):
    """Used for Multi-Class (e.g., MNIST Digits 0-9)."""
    y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
    return -np.sum(y_true * np.log(y_pred)) / y_true.shape[0]

def hinge_loss(y_true, y_pred):
    """Used for SVMs. Max(0, 1 - y_true * y_pred)."""
    return np.mean(np.maximum(0, 1 - y_true * y_pred))

if __name__ == "__main__":
    # Test block to verify everything works
    y_real = np.array([[1.0, 0.0, 1.0]])
    y_guess = np.array([[0.9, 0.1, 0.4]]) 

    print(f"Computed Cost (train.py style): {compute_cost(y_guess, y_real):.4f}")
    print(f"MAE Loss: {mae(y_real, y_guess):.4f}")