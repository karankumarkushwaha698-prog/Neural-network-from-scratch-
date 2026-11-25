
# ============================================================
# PROJECT 1: NEURAL NETWORK FROM SCRATCH - XOR CLASSIFIER
# Built using Python + NumPy (no PyTorch)
# ============================================================

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# ------------------------------------------------------------
# 1. DATASET
# ------------------------------------------------------------

X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
], dtype=float)

y = np.array([
    [0],
    [1],
    [1],
    [0]
], dtype=float)

print("XOR Dataset:")
print("Inputs:\n", X)
print("Targets:\n", y)


# ------------------------------------------------------------
# 2. PERCEPTRON (A SINGLE NEURON)
# ------------------------------------------------------------

class Perceptron:

    def __init__(self, input_size, learning_rate=0.1):
        self.weights = np.zeros(input_size)
        self.bias = 0.0
        self.learning_rate = learning_rate

    def predict(self, x):
        z = np.dot(x, self.weights) + self.bias
        return 1 if z >= 0 else 0

    def train(self, X, y, epochs=100):
        for epoch in range(epochs):

            mistakes = 0

            for x, target in zip(X, y.flatten()):
                prediction = self.predict(x)

                error = target - prediction

                # Perceptron learning rule
                self.weights += self.learning_rate * error * x
                self.bias += self.learning_rate * error

                if error != 0:
                    mistakes += 1

            if epoch % 20 == 0:
                print(f"Perceptron Epoch {epoch}, mistakes: {mistakes}")


print("\n--- Training a Single Perceptron ---")

perceptron = Perceptron(input_size=2)
perceptron.train(X, y)

print("\nPerceptron predictions:")
for x in X:
    print(x, "->", perceptron.predict(x))


# ------------------------------------------------------------
# 3. ACTIVATION FUNCTIONS
# ------------------------------------------------------------

def sigmoid(z):
    # Clip values to avoid overflow
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))


def sigmoid_derivative_from_output(a):
    return a * (1 - a)


def tanh_derivative_from_output(a):
    return 1 - a**2


# ------------------------------------------------------------
# 4. NEURAL NETWORK
#    Architecture: 2 inputs -> 8 hidden neurons -> 1 output
# ------------------------------------------------------------

class NeuralNetwork:

    def __init__(self, input_size, hidden_size, output_size,
                 learning_rate=0.1):

        self.learning_rate = learning_rate

        # Initialize weights
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(1 / input_size)
        self.b1 = np.zeros((1, hidden_size))

        self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(1 / hidden_size)
        self.b2 = np.zeros((1, output_size))

    # --------------------------------------------------------
    # 5. FORWARD PROPAGATION
    # --------------------------------------------------------

    def forward(self, X):

        # Input -> Hidden layer
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = np.tanh(self.z1)

        # Hidden layer -> Output layer
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = sigmoid(self.z2)

        return self.a2

    # --------------------------------------------------------
    # 6. LOSS FUNCTION (BINARY CROSS-ENTROPY)
    # --------------------------------------------------------

    def compute_loss(self, y_true, y_pred):

        epsilon = 1e-8
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

        loss = -np.mean(
            y_true * np.log(y_pred) +
            (1 - y_true) * np.log(1 - y_pred)
        )

        return loss

    # --------------------------------------------------------
    # 7. BACKPROPAGATION
    # --------------------------------------------------------

    def backward(self, X, y):

        m = X.shape[0]

        # Output layer gradient
        dz2 = self.a2 - y

        dW2 = np.dot(self.a1.T, dz2) / m
        db2 = np.sum(dz2, axis=0, keepdims=True) / m

        # Hidden layer gradient
        dz1 = np.dot(dz2, self.W2.T) * tanh_derivative_from_output(self.a1)

        dW1 = np.dot(X.T, dz1) / m
        db1 = np.sum(dz1, axis=0, keepdims=True) / m

        # Update weights and biases using gradient descent
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2

        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1

    # --------------------------------------------------------
    # 8. TRAINING LOOP
    # --------------------------------------------------------

    def train(self, X, y, epochs=12000):

        self.loss_history = []

        for epoch in range(epochs):

            # Forward pass
            predictions = self.forward(X)

            # Calculate loss
            loss = self.compute_loss(y, predictions)
            self.loss_history.append(loss)

            # Backward pass
            self.backward(X, y)

            if epoch % 2000 == 0:
                print(f"Epoch: {epoch:5d} | Loss: {loss:.6f}")

    # --------------------------------------------------------
    # 9. PREDICTION
    # --------------------------------------------------------

    def predict(self, X):

        probabilities = self.forward(X)

        return (probabilities >= 0.5).astype(int)


# ------------------------------------------------------------
# 10. CREATE AND TRAIN THE NETWORK
# ------------------------------------------------------------

print("\n--- Training Neural Network ---")

model = NeuralNetwork(
    input_size=2,
    hidden_size=8,
    output_size=1,
    learning_rate=0.1
)

model.train(X, y, epochs=12000)


# ------------------------------------------------------------
# 11. TEST THE MODEL
# ------------------------------------------------------------

print("\n--- Final XOR Results ---")

probabilities = model.forward(X)
predictions = (probabilities >= 0.5).astype(int)

for i in range(len(X)):
    print(
        f"Input: {X[i]} | "
        f"Actual: {int(y[i][0])} | "
        f"Probability: {probabilities[i][0]:.4f} | "
        f"Predicted: {predictions[i][0]}"
    )

accuracy = np.mean(predictions == y) * 100

print(f"\nAccuracy: {accuracy:.2f}%")
print(f"Final Loss: {model.loss_history[-1]:.6f}")


# ------------------------------------------------------------
# 12. LOSS GRAPH
# ------------------------------------------------------------

plt.figure(figsize=(8, 4))
plt.plot(model.loss_history)
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Binary Cross-Entropy")
plt.grid(True)
plt.show()


# ------------------------------------------------------------
# 13. DECISION BOUNDARY VISUALIZATION
# ------------------------------------------------------------

grid_x = np.linspace(-0.2, 1.2, 200)
grid_y = np.linspace(-0.2, 1.2, 200)

xx, yy = np.meshgrid(grid_x, grid_y)

grid = np.c_[xx.ravel(), yy.ravel()]

grid_probabilities = model.forward(grid)
Z = grid_probabilities.reshape(xx.shape)

plt.figure(figsize=(7, 6))

plt.contourf(
    xx, yy, Z,
    levels=50,
    cmap="cool",
    alpha=0.8
)

plt.contour(
    xx, yy, Z,
    levels=[0.5],
    colors="black",
    linewidths=2
)

for i in range(len(X)):
    if y[i][0] == 0:
        plt.scatter(X[i, 0], X[i, 1], marker="s",
                   color="blue", s=100, label="Class 0" if i == 0 else "")
    else:
        plt.scatter(X[i, 0], X[i, 1], marker="^",
                   color="red", s=100, label="Class 1" if i == 1 else "")

plt.title("Neural Network XOR Decision Boundary")
plt.xlabel("Input 1")
plt.ylabel("Input 2")
plt.legend()
plt.grid(True)
plt.show()

print("\nProject completed successfully!")
