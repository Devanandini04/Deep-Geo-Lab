import numpy as np

class Perceptron:
    def __init__(self, input_size):
        # Initialize weights (W) and Bias (b) with zeros
        # We need 1 weight per input.
        self.weights = np.zeros(input_size)
        self.bias = 0
        self.learning_rate = 0.1

    def activation(self, z):
        # Step Function: If z > 0, return 1. Else return 0.
        return 1 if z > 0 else 0

    def predict(self, inputs):
        # Formula: Z = (W . X) + b
        z = np.dot(inputs, self.weights) + self.bias
        return self.activation(z)

    def train(self, X, y, epochs=10):
        # The Learning Loop (Gradient Descent Logic)
        print(f"Starting weights: {self.weights}")
        
        for epoch in range(epochs):
            for i in range(len(X)):
                inputs = X[i]
                target = y[i]
                
                # 1. Ask model to guess
                prediction = self.predict(inputs)
                
                # 2. Calculate Error (Target - Prediction)
                error = target - prediction
                
                # 3. Update Weights (Math: W_new = W_old + lr * error * input)
                self.weights += self.learning_rate * error * inputs
                self.bias += self.learning_rate * error

# --- Main Execution ---
if __name__ == "__main__":
    # DATA: The "AND" Gate Logic
    # [0, 0] -> 0
    # [0, 1] -> 0
    # [1, 0] -> 0
    # [1, 1] -> 1 (Only fires here)
    X = np.array([[0,0], [0,1], [1,0], [1,1]])
    y = np.array([0, 0, 0, 1])

    # Initialize Neuron
    neuron = Perceptron(input_size=2)
    
    # Train it
    neuron.train(X, y, epochs=10)
    
    print("-----------------------")
    print(f"Learned Weights: {neuron.weights}")
    print(f"Learned Bias: {neuron.bias}")
    
    # Test it
    test_input = np.array([1, 1])
    print(f"Testing [1, 1]: {neuron.predict(test_input)}")