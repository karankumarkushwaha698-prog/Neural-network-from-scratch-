
# ============================================================
# XOR NEURAL NETWORK - INTERACTIVE WEB APPLICATION
# Python + NumPy + Streamlit
# ============================================================

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# PAGE CONFIGURATION
# ------------------------------------------------------------

st.set_page_config(
    page_title="XOR Neural Network",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 Neural Network from Scratch")
st.subheader("XOR Classification using NumPy")

st.write(
    "An interactive demonstration of a neural network "
    "trained using forward propagation, backpropagation "
    "and gradient descent."
)

# ------------------------------------------------------------
# DATASET
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


# ------------------------------------------------------------
# ACTIVATION FUNCTIONS
# ------------------------------------------------------------

def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))


def tanh_derivative(a):
    return 1 - a**2


# ------------------------------------------------------------
# NEURAL NETWORK CLASS
# ------------------------------------------------------------

class NeuralNetwork:

    def __init__(self, input_size, hidden_size,
                 output_size, learning_rate):

        self.learning_rate = learning_rate

        self.W1 = (
            np.random.randn(input_size, hidden_size)
            * np.sqrt(1 / input_size)
        )

        self.b1 = np.zeros((1, hidden_size))

        self.W2 = (
            np.random.randn(hidden_size, output_size)
            * np.sqrt(1 / hidden_size)
        )

        self.b2 = np.zeros((1, output_size))

    # Forward propagation
    def forward(self, X):

        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = np.tanh(self.z1)

        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = sigmoid(self.z2)

        return self.a2

    # Loss calculation
    def compute_loss(self, y_true, y_pred):

        epsilon = 1e-8

        y_pred = np.clip(
            y_pred,
            epsilon,
            1 - epsilon
        )

        return -np.mean(
            y_true * np.log(y_pred)
            + (1 - y_true) * np.log(1 - y_pred)
        )

    # Backpropagation
    def backward(self, X, y):

        m = X.shape[0]

        dz2 = self.a2 - y

        dW2 = np.dot(self.a1.T, dz2) / m
        db2 = np.sum(dz2, axis=0, keepdims=True) / m

        dz1 = (
            np.dot(dz2, self.W2.T)
            * tanh_derivative(self.a1)
        )

        dW1 = np.dot(X.T, dz1) / m
        db1 = np.sum(dz1, axis=0, keepdims=True) / m

        # Gradient descent updates
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2

        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1

    # Training
    def train(self, X, y, epochs):

        self.loss_history = []

        for epoch in range(epochs):

            predictions = self.forward(X)

            loss = self.compute_loss(y, predictions)

            self.loss_history.append(loss)

            self.backward(X, y)

    # Prediction
    def predict(self, X):

        probabilities = self.forward(X)

        return (probabilities >= 0.5).astype(int)


# ------------------------------------------------------------
# TRAIN THE MODEL
# ------------------------------------------------------------

@st.cache_resource
def train_model():

    np.random.seed(42)

    model = NeuralNetwork(
        input_size=2,
        hidden_size=8,
        output_size=1,
        learning_rate=0.1
    )

    model.train(X, y, epochs=12000)

    return model


with st.spinner("Training neural network..."):
    model = train_model()

st.success("Neural network training completed!")


# ------------------------------------------------------------
# MODEL INFORMATION
# ------------------------------------------------------------

st.header("Model Architecture")

col1, col2, col3 = st.columns(3)

col1.metric("Input Neurons", "2")
col2.metric("Hidden Neurons", "8")
col3.metric("Output Neurons", "1")

st.write("**Activation functions:** Tanh and Sigmoid")
st.write("**Loss function:** Binary Cross-Entropy")
st.write("**Optimizer:** Gradient Descent")


# ------------------------------------------------------------
# INTERACTIVE PREDICTION
# ------------------------------------------------------------

st.header("Try the XOR Classifier")

col1, col2 = st.columns(2)

with col1:
    input_1 = st.selectbox(
        "Select Input 1",
        [0, 1]
    )

with col2:
    input_2 = st.selectbox(
        "Select Input 2",
        [0, 1]
    )

user_input = np.array(
    [[input_1, input_2]],
    dtype=float
)

probability = model.forward(user_input)[0][0]

prediction = model.predict(user_input)[0][0]

st.subheader("Prediction Result")

col1, col2 = st.columns(2)

col1.metric(
    "Predicted Class",
    int(prediction)
)

col2.metric(
    "Prediction Probability",
    f"{probability * 100:.2f}%"
)

st.write(
    "Classification:",
    "Class 1" if prediction == 1 else "Class 0"
)


# ------------------------------------------------------------
# COMPLETE XOR TEST
# ------------------------------------------------------------

st.header("Complete XOR Test")

all_probabilities = model.forward(X)
all_predictions = model.predict(X)

results = []

for i in range(len(X)):

    results.append({
        "Input 1": int(X[i][0]),
        "Input 2": int(X[i][1]),
        "Actual Output": int(y[i][0]),
        "Predicted Output": int(all_predictions[i][0]),
        "Probability": round(
            float(all_probabilities[i][0]), 4
        )
    })

st.dataframe(
    results,
    use_container_width=True
)

accuracy = np.mean(all_predictions == y) * 100

st.metric("Training Dataset Accuracy", f"{accuracy:.2f}%")

st.metric(
    "Final Training Loss",
    f"{model.loss_history[-1]:.6f}"
)


# ------------------------------------------------------------
# LOSS GRAPH
# ------------------------------------------------------------

st.header("Training Loss Visualization")

fig1, ax1 = plt.subplots(figsize=(9, 4))

ax1.plot(model.loss_history)

ax1.set_title("Neural Network Training Loss")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Binary Cross-Entropy Loss")
ax1.grid(True)

st.pyplot(fig1)

plt.close(fig1)


# ------------------------------------------------------------
# DECISION BOUNDARY
# ------------------------------------------------------------

st.header("Decision Boundary Visualization")

grid_x = np.linspace(-0.2, 1.2, 200)
grid_y = np.linspace(-0.2, 1.2, 200)

xx, yy = np.meshgrid(grid_x, grid_y)

grid = np.c_[
    xx.ravel(),
    yy.ravel()
]

grid_probabilities = model.forward(grid)

Z = grid_probabilities.reshape(xx.shape)

fig2, ax2 = plt.subplots(figsize=(8, 6))

contour = ax2.contourf(
    xx,
    yy,
    Z,
    levels=50,
    cmap="cool",
    alpha=0.8
)

fig2.colorbar(
    contour,
    ax=ax2,
    label="Probability of Class 1"
)

ax2.contour(
    xx,
    yy,
    Z,
    levels=[0.5],
    colors="black",
    linewidths=2
)

for i in range(len(X)):

    if y[i][0] == 0:

        ax2.scatter(
            X[i, 0],
            X[i, 1],
            marker="s",
            color="blue",
            s=100
        )

    else:

        ax2.scatter(
            X[i, 0],
            X[i, 1],
            marker="^",
            color="red",
            s=100
        )

ax2.set_title("XOR Neural Network Decision Boundary")
ax2.set_xlabel("Input 1")
ax2.set_ylabel("Input 2")

ax2.grid(True)

st.pyplot(fig2)

plt.close(fig2)


# ------------------------------------------------------------
# FOOTER
# ------------------------------------------------------------

st.divider()

st.write(
    "Built with Python, NumPy, Matplotlib and Streamlit."
)

st.caption(
    "Educational project: Neural Network from Scratch - XOR Classifier"
)
