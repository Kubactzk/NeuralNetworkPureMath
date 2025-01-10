import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns


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
    plt.savefig("data.jpg")
    plt.close()
    

    return df

# Xavier initialization
def init_weights(input_size, hidden_size, output_size):
    w1 = np.random.randn(hidden_size, input_size) * np.sqrt(2 / (input_size + hidden_size))
    b1 = np.zeros((hidden_size, 1))
    w2 = np.random.randn(output_size, hidden_size) * np.sqrt(2 / (hidden_size + output_size))
    b2 = np.zeros((output_size, 1))
    return w1, b1, w2, b2

# Sigmoid activation function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Derivative of sigmoid
def sigmoid_derivative(a):
    return a * (1 - a)

# Mean Squared Error loss
def compute_loss(y, a2):
    return np.mean((y - a2) ** 2)

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
        w1_arr.append(w1.copy())  # Copy the weights
        w2_arr.append(w2.copy())

        # Forward pass
        _, a1, _, a2 = forward_pass(w1, b1, w2, b2, x)
        

        # Compute loss
        loss = compute_loss(y, a2)
        losses.append(loss)

        if earlyStopping(loss, 0.05):
            print(f"Early stopping at epoch {epoch}, Loss: {loss:.4f}")
            break

        # Backward pass
        w1, b1, w2, b2 = backward_pass(w1, b1, w2, b2, x, y, a1, a2, lr)

        # Print loss every 1000 epochs
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss:.4f}")
        

    return w1, b1, w2, b2, losses, w1_arr, w2_arr

# Model evaluation
def evaluate_model(w1, b1, w2, b2, x, y):
    _, _, _, a2 = forward_pass(w1, b1, w2, b2, x)
    predictions = (a2 > 0.5).astype(int).flatten()
    accuracy = accuracy_score(y, predictions)
    cm = confusion_matrix(y, predictions)
    return accuracy, cm, predictions



def plot_weights(w1, w2):
    import seaborn as sns
    import matplotlib.pyplot as plt
    
    # Initialize weight values and epochs
    w1_00, w1_01, w1_10, w1_11 = [], [], [], []
    w2_00, w2_01 = [], []
    epochs = []
    
    # Extract weights and epochs
    for i in range(len(w1)):
        epochs.append(i + 1)
        w1_00.append(w1[i][0][0])
        w1_01.append(w1[i][0][1])
        w1_10.append(w1[i][1][0])
        w1_11.append(w1[i][1][1])
        w2_00.append(w2[i][0][0])
        w2_01.append(w2[i][0][1])
    
    # Data dictionaries
    weights = {"w00": w1_00, "w01": w1_01, "w10": w1_10, "w11": w1_11}
    weights2 = {"w200": w2_00, "w201": w2_01}
    
    # Set a professional style
    sns.set_theme(style="whitegrid")
    palette = sns.color_palette("husl", len(weights))  # Distinct colors for Layer 1

    # Plot Layer 1 weights
    plt.figure(figsize=(10, 6))
    for i, (weight_name, values) in enumerate(weights.items()):
        plt.plot(epochs, values, label=weight_name, marker='o', linestyle='-', color=palette[i])
        
        # Annotate visible points
        x_ticks = plt.xticks()[0]  # Current x-axis tick positions
        visible_indices = [int(tick - 1) for tick in x_ticks if 1 <= tick <= len(values)]
        for idx in visible_indices:
            plt.text(
                epochs[idx], values[idx], f'{values[idx]:.2f}', 
                fontsize=10, fontweight='bold', color='black', ha='center', va='bottom'
            )
    
    plt.title("Layer 1 Weights Over Time", fontsize=16)
    plt.xlabel("Epochs", fontsize=14)
    plt.ylabel("Weight Values", fontsize=14)
    plt.legend(title="Weights", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig("w1_annotated.jpg", dpi=300)
    plt.show()
    plt.close()

    # Plot Layer 2 weights
    plt.figure(figsize=(10, 6))
    for i, (weight_name, values) in enumerate(weights2.items()):
        plt.plot(epochs, values, label=weight_name, marker='s', linestyle='--', color=palette[i])
        
        # Annotate visible points
        x_ticks = plt.xticks()[0]  # Current x-axis tick positions
        visible_indices = [int(tick - 1) for tick in x_ticks if 1 <= tick <= len(values)]
        for idx in visible_indices:
            plt.text(
                epochs[idx], values[idx], f'{values[idx]:.2f}', 
                fontsize=10, fontweight='bold', color='black', ha='center', va='bottom'
            )
    
    plt.title("Layer 2 Weights Over Time", fontsize=16)
    plt.xlabel("Epochs", fontsize=14)
    plt.ylabel("Weight Values", fontsize=14)
    plt.legend(title="Weights", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig("w2_annotated.jpg", dpi=300)
    plt.show()
    plt.close()





def plotLoss(losses):
    # Set a professional style
    sns.set_theme(style="whitegrid")
    
    # Plot the loss
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(losses) + 1)  # Create epoch range
    plt.plot(epochs, losses, marker='o', markersize=6, linestyle='-', linewidth=2, color='blue', label="Loss")
    
    # Get visible x-axis ticks
    x_ticks = plt.xticks()[0]  # Current x-axis tick positions
    visible_indices = [int(tick - 1) for tick in x_ticks if 1 <= tick <= len(losses)]
    
    # Annotate data points only at x-axis tick positions
    for idx in visible_indices:
        plt.text(
            epochs[idx], losses[idx], f'{losses[idx]:.2f}', 
            fontsize=12, fontweight='bold', color='black', ha='center', va='bottom'
        )
    
    # Add grid, labels, and title
    plt.title("Loss Over Epochs", fontsize=18)
    plt.xlabel("Epochs", fontsize=14)
    plt.ylabel("Loss", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Tight layout and save the plot
    plt.tight_layout()
    plt.savefig("MSE_Loss_Visible.jpg", dpi=300, bbox_inches='tight')  # High-quality image
    plt.show()
    plt.close()



if __name__ == "__main__":
    # Generate dataset
    data = dataset()
    train_data, test_data = train_test_split(data, test_size=0.3)

    # Extract features and labels
    x_train = train_data[['x1', 'x2']].values
    y_train = train_data['y'].values
    x_test = test_data[['x1', 'x2']].values
    y_test = test_data['y'].values

    # train the model
    w1, b1, w2, b2, losses, w1_arr, w2_arr = train(x_train, y_train, input_size=2, hidden_size=2, output_size=1, epochs=100000, lr=0.05)

    #plot loss
    plotLoss(losses)

    #plot weights
    plot_weights(w1_arr, w2_arr)
    # Evaluate the model
    accuracy, cm, predictions = evaluate_model(w1, b1, w2, b2, x_test, y_test)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")
    print("Confusion Matrix:")
    print(cm)

    # Display a few predictions
    test_sample = 10
    for i in range(test_sample):
        print(f"Input: {x_test[i]}, Prediction: {predictions[i]}, True: {y_test[i]}")
