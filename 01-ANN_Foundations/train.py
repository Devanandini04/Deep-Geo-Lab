import numpy as np

# --- IMPORTING YOUR MODULES ---
# Ensure these filenames match exactly what you have in your folder
from mlp_forward import forward_propagation
from mlp_backward import backward_propagation
from loss_functions import compute_cost
from optimizer import update_parameters

def initialize_parameters(n_x, n_h, n_y):
    """
    Initializes weight matrices and bias vectors.
    
    Arguments:
    n_x -- size of input layer (number of features)
    n_h -- size of hidden layer (number of neurons)
    n_y -- size of output layer (number of classes)
    
    Returns:
    parameters -- python dictionary containing:
                  W1: weight matrix of shape (n_h, n_x)
                  b1: bias vector of shape (n_h, 1)
                  W2: weight matrix of shape (n_y, n_h)
                  b2: bias vector of shape (n_y, 1)
    """
    np.random.seed(42) # Fixed seed for reproducibility
    
    # Initialize weights with small random values to break symmetry
    # Scale by 0.01 to keep values small (helps with convergence)
    W1 = np.random.randn(n_h, n_x) * 0.01
    b1 = np.zeros((n_h, 1))
    
    W2 = np.random.randn(n_y, n_h) * 0.01
    b2 = np.zeros((n_y, 1))
    
    parameters = {"W1": W1, "b1": b1, "W2": W2, "b2": b2}
    return parameters

def train_model(X, Y, n_h, num_iterations=10000, learning_rate=0.01):
    """
    The main driver function that runs the training loop (Gradient Descent).
    
    Arguments:
    X -- input data dataset of shape (n_x, number of examples)
    Y -- labels of shape (n_y, number of examples)
    n_h -- size of the hidden layer
    num_iterations -- number of iterations in gradient descent loop
    learning_rate -- learning rate of the gradient descent update rule
    
    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    """
    np.random.seed(42)
    n_x = X.shape[0]
    n_y = Y.shape[0]
    
    # 1. Initialize Parameters
    parameters = initialize_parameters(n_x, n_h, n_y)
    
    # 2. Gradient Descent Loop
    for i in range(0, num_iterations):
         
        # A. Forward Propagation (Predict)
        # Returns A2 (prediction) and cache (stored values for backprop)
        A2, cache = forward_propagation(X, parameters)
        
        # B. Compute Cost (Measure Error)
        cost = compute_cost(A2, Y)
 
        # C. Backward Propagation (Calculate Gradients)
        grads = backward_propagation(A2, Y, cache)
 
        # D. Update Parameters (Optimize)
        parameters = update_parameters(parameters, grads, learning_rate)
        
        # Print the cost every 1000 iterations to monitor progress
        if i % 1000 == 0:
            print(f"Cost after iteration {i}: {cost:.6f}")
            
    return parameters

# --- MAIN EXECUTION BLOCK ---
# This block only runs if you execute this file directly.
if __name__ == "__main__":
    print("--- STARTING TRAINING ON XOR DATASET ---")

    # 1. PREPARE DATA (XOR Logic Gate)
    # Input X: (2 features, 4 examples) -> Transposed form
    # Features: [0,0], [0,1], [1,0], [1,1]
    X = np.array([[0, 0, 1, 1], [0, 1, 0, 1]])

    # Labels Y: (1 output, 4 examples)
    # Target: 0, 1, 1, 0
    Y = np.array([[0, 1, 1, 0]])

    # 2. TRAIN THE MODEL
    # We use 4 neurons in the hidden layer
    # We use a higher learning rate (1.2) for this simple problem
    trained_params = train_model(X, Y, n_h=4, num_iterations=10000, learning_rate=1.2)

    # 3. TEST PREDICTIONS
    print("\n--- FINAL RESULTS ---")
    predictions, _ = forward_propagation(X, trained_params)
    
    print("Target Labels (Y):", Y)
    print("Model Predictions:", predictions)
    print("Rounded Predictions:", np.round(predictions))