import numpy as np

# --- 1. Define the Activation Function ---
# The Sigmoid function squashes any number to be between 0 and 1.
# We need this to introduce non-linearity (so the model can learn curves, not just straight lines).
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class MLP:
    def __init__(self, input_size, hidden_size, output_size):
        """
        Building the Architecture (The Skeleton)
        Args:
            input_size: Number of features in input data 
            hidden_size: Number of neurons in the hidden layer
            output_size: Number of predictions (e.g., 1 for Yes/No)
        """
        print(f"Initializing MLP: {input_size} Inputs -> {hidden_size} Hidden -> {output_size} Output")
        
        # --- Weight Initialization ---
        # W1: Weights connecting Input to Hidden Layer
        # Shape: (input_size, hidden_size)
        self.W1 = np.random.randn(input_size, hidden_size)
        self.b1 = np.zeros((1, hidden_size)) # Bias for hidden layer
        
        # W2: Weights connecting Hidden to Output Layer
        # Shape: (hidden_size, output_size)
        self.W2 = np.random.randn(hidden_size, output_size)
        self.b2 = np.zeros((1, output_size)) # Bias for output layer

    def forward(self, X):
        """
        The "Forward Propagation" Step.
        Data flows from Input -> Hidden -> Output.
        """
        # --- Layer 1: Input -> Hidden ---
        # 1. Linear Transformation (Z = X.W + b)
        self.z1 = np.dot(X, self.W1) + self.b1
        
        # 2. Activation (Non-linear)
        self.a1 = sigmoid(self.z1)
        
        # --- Layer 2: Hidden -> Output ---
        # 3. Linear Transformation
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        
        # 4. Final Activation (Prediction)
        self.output = sigmoid(self.z2)
        
        return self.output

# --- Main Execution (Testing the Class) ---
if __name__ == "__main__":
    # Example: A Satellite Pixel with 3 bands (Red, Green, Blue)
    # Let's say the values are [0.5, 0.8, 0.1]
    input_data = np.array([[0.5, 0.8, 0.1]])
    
    # Initialize the Network
    # 3 Inputs -> 5 Hidden Neurons -> 1 Output (Is it Water?)
    model = MLP(input_size=3, hidden_size=5, output_size=1)
    
    # Pass the data through the network
    prediction = model.forward(input_data)
    
    print("\n-------------------------")
    print(f"Input Data: {input_data}")
    print(f"Model Prediction: {prediction}")
    print("-------------------------")
    print("Note: The prediction is random because we haven't trained the weights yet!")