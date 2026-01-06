import numpy as np

def sigmoid(Z):
    """
    Computes the sigmoid of Z
    """
    return 1 / (1 + np.exp(-Z))

def relu(Z):
    """
    Computes the ReLU of Z (Linear for positive values, 0 for negative)
    """
    return np.maximum(0, Z)

def forward_propagation(X, parameters):
    """
    Argument:
    X -- input data of size (n_x, m)
    parameters -- python dictionary containing your parameters (W1, b1, W2, b2)
    
    Returns:
    A2 -- The sigmoid output of the second activation
    cache -- a dictionary containing "Z1", "A1", "Z2" and "A2"
    """
    # Retrieve parameters from the dictionary
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    
    # --- Layer 1 (Input -> Hidden) ---
    # Z1 = W1 . X + b1
    Z1 = np.dot(W1, X) + b1
    # Activation Function: ReLU (Standard for hidden layers)
    A1 = relu(Z1)
    
    # --- Layer 2 (Hidden -> Output) ---
    # Z2 = W2 . A1 + b2
    Z2 = np.dot(W2, A1) + b2
    # Activation Function: Sigmoid (Standard for binary output)
    A2 = sigmoid(Z2)
    
    # Compatibility check
    assert(A2.shape == (1, X.shape[1]))
    
    # Cache values needed for Backward Propagation
    cache = (Z1, A1, W2, X)
    
    return A2, cache