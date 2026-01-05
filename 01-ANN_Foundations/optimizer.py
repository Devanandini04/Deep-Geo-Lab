import numpy as np

def update_parameters(parameters, grads, learning_rate=0.01):
    """
    Gradient Descent ka use karke parameters (Weights aur Biases) ko update kar rahe h.
    
    Arguments:
    parameters -- Python dictionary jahan current weights aur biases hain (e.g., "W1", "b1").
    grads -- Python dictionary jahan gradients (slope) hain, jo backward_propagation se calculate hoke aaye hain.
    learning_rate -- Ek scalar value (step size).
    
    Returns:
    parameters -- Python dictionary jahan ab updated (nayi) values aayegi!
    """
    L = len(parameters) // 2 
    for l in range(1, L + 1):
        parameters["W" + str(l)] = parameters["W" + str(l)] - learning_rate * grads["dW" + str(l)]
        parameters["b" + str(l)] = parameters["b" + str(l)] - learning_rate * grads["db" + str(l)]
        
    return parameters