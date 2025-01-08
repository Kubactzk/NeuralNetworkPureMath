import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

# Generate XOR dataset
def plot_weights(w1, w2):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plot weights of the first layer
    im1 = axes[0].imshow(w1, cmap='coolwarm', aspect='auto')
    axes[0].set_title("Weights: Input to Hidden Layer")
    axes[0].set_xlabel("Input Neurons")
    axes[0].set_ylabel("Hidden Neurons")
    fig.colorbar(im1, ax=axes[0])

    # Plot weights of the second layer
    im2 = axes[1].imshow(w2, cmap='coolwarm', aspect='auto')
    axes[1].set_title("Weights: Hidden to Output Layer")
    axes[1].set_xlabel("Hidden Neurons")
    axes[1].set_ylabel("Output Neurons")
    fig.colorbar(im2, ax=axes[1])

    plt.tight_layout()
    plt.show()


def dataset():
    df = pd.DataFrame([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 0]] * 1000)
    df.columns = ['x1', 'x2', 'y']
    df['x1'] += np.random.normal(scale=0.1, size=4000)  # Reduced noise
    df['x2'] += np.random.normal(scale=0.1, size=4000)

    # Visualize data
    plt.scatter(df['x1'], df['x2'], c=df['y'], cmap='coolwarm', alpha=0.6, edgecolor='k')
    plt.title("Data Distribution")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.colorbar(label="y")
    plt.show()

    return df

# Sigmoid activation function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Derivative of sigmoid
def sigmoid_derivative(a):
    return a * (1 - a)

# Mean Squared Error loss
def compute_loss(y, a2):
    return np.mean((y - a2) ** 2)

# Xavier initialization
def init_weights(input_size, hidden_size, output_size):
    w1 = np.random.randn(hidden_size, input_size) * np.sqrt(2 / (input_size + hidden_size))
    b1 = np.zeros((hidden_size, 1))
    w2 = np.random.randn(output_size, hidden_size) * np.sqrt(2 / (hidden_size + output_size))
    b2 = np.zeros((output_size, 1))
    return w1, b1, w2, b2

# Forward pass
def forward_pass(w1, b1, w2, b2, x):
    z1 = np.dot(w1, x.T) + b1
    a1 = sigmoid(z1)
    z2 = np.dot(w2, a1) + b2
    a2 = sigmoid(z2)
    return z1, a1, z2, a2

# Backward pass
def backward_pass(w1, b1, w2, b2, x, y, a1, a2, lr):
    m = y.shape[0]

    dz2 = a2 - y.reshape(1, -1)  # Gradient of output layer
    dw2 = np.dot(dz2, a1.T) / m
    db2 = np.sum(dz2, axis=1, keepdims=True) / m

    dz1 = np.dot(w2.T, dz2) * sigmoid_derivative(a1)
    dw1 = np.dot(dz1, x) / m
    db1 = np.sum(dz1, axis=1, keepdims=True) / m

    # Update weights and biases
    w1 -= lr * dw1
    b1 -= lr * db1
    w2 -= lr * dw2
    b2 -= lr * db2

    return w1, b1, w2, b2

def earlyStopping(loss, treshold):
    return loss < treshold

# Training function
def train(x, y, input_size, hidden_size, output_size, epochs=10000, lr=0.005):
    w1_arr = []
    w2_arr = []
    w1, b1, w2, b2 = init_weights(input_size, hidden_size, output_size)
    losses = []

    for epoch in range(epochs):
        # Forward pass
        _, a1, _, a2 = forward_pass(w1, b1, w2, b2, x)

        # Compute loss
        loss = compute_loss(y, a2)
        losses.append(loss)

        if earlyStopping(loss, 0.1):
            print(f"Early stopping at epoch {epoch}, Loss: {loss:.4f}")
            break

        # Backward pass
        w1, b1, w2, b2 = backward_pass(w1, b1, w2, b2, x, y, a1, a2, lr)

        # Print loss every 1000 epochs
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss:.4f}")

    return w1, b1, w2, b2, losses

# Model evaluation
def evaluate_model(w1, b1, w2, b2, x, y):
    _, _, _, a2 = forward_pass(w1, b1, w2, b2, x)
    predictions = (a2 > 0.5).astype(int).flatten()
    accuracy = accuracy_score(y, predictions)
    cm = confusion_matrix(y, predictions)
    return accuracy, cm, predictions

if __name__ == "__main__":
    # Generate dataset
    data = dataset()
    train_data, test_data = train_test_split(data, test_size=0.3)

    # Extract features and labels
    x_train = train_data[['x1', 'x2']].values
    y_train = train_data['y'].values
    x_test = test_data[['x1', 'x2']].values
    y_test = test_data['y'].values

    # Train the model
    w1, b1, w2, b2, losses = train(x_train, y_train, input_size=2, hidden_size=2, output_size=1, epochs=10000, lr=0.05)

    # Plot the loss
    plt.plot(losses)
    plt.title("Loss over epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.show()

    # Evaluate the model
    accuracy, cm, predictions = evaluate_model(w1, b1, w2, b2, x_test, y_test)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")
    print("Confusion Matrix:")
    print(cm)

    # Display a few predictions
    test_sample = 10
    for i in range(test_sample):
        print(f"Input: {x_test[i]}, Prediction: {predictions[i]}, True: {y_test[i]}")
