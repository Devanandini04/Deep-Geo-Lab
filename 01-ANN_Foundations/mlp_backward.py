import numpy as np

def backward_propagation(Y_hat, Y, cache):
    """
    Computes the gradients for a 2-layer MLP.
    
    Arguments:
    Y_hat -- predicted output (A2) from forward pass
    Y -- true label vector
    cache -- dictionary containing (Z1, A1, W2, X) stored during forward pass
    
    Returns:
    grads -- dictionary containing gradients dW1, db1, dW2, db2
    """
    m = Y.shape[1]
    (Z1, A1, W2, X) = cache
    
    # 1. Output Layer Gradients
    # Assuming Binary Cross Entropy Loss and Sigmoid Activation for output
    dZ2 = Y_hat - Y
    
    dW2 = (1 / m) * np.dot(dZ2, A1.T)
    db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)
    
    # 2. Hidden Layer Gradients
    # Backpropagate to hidden layer
    dA1 = np.dot(W2.T, dZ2)
    
    # Derivative of ReLU activation function (assuming ReLU in hidden layer)
    dZ1 = np.array(dA1, copy=True)
    dZ1[Z1 <= 0] = 0 
    
    dW1 = (1 / m) * np.dot(dZ1, X.T)
    db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)
    
    grads = {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}
    
    return grads