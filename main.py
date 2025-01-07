import numpy as np
import jax.numpy as jnp
import pandas as pd

data = pd.read_csv('venv/data/train.csv')


# set up data

data = jnp.array(data)
n, d = data.shape
d -= 1
np.random.shuffle(data)

data_dev = data[0:1000]
data_train = data[1000:]

# separate features (X) and labels (Y)
# all columns except the first one (features)
X_dev = data_dev[:, 1:]
X_dev = X_dev / 255
X_train = data_train[:, 1:]
X_train = X_train / 255

# first column (labels)
Y_dev = data_dev[:, 0]
Y_train = data_train[:, 0]


# model
def init_params():
    # random values between -0.5 to 0.5
    W1 = np.random.rand(10, 784) - 0.5
    W2 = np.random.rand(10, 10) - 0.5
    W3 = np.random.rand(10, 10) - 0.5

    b1 = np.random.rand(1, 10) - 0.5
    b2 = np.random.rand(1, 10) - 0.5
    b3 = np.random.rand(1, 10) - 0.5
    return W1, b1, W2, b2, W3, b3

def relu(Z):
    return np.maximum(0, Z)

def softmax(Z):
    return np.exp(Z) / np.sum(np.exp(Z), axis=1, keepdims=True)

def forward_prop(W1, b1, W2, b2, W3, b3, X):
    # first hidden layer
    Z1 = X.dot(W1.T) + b1
    A1 = relu(Z1)

    # second hidden layer
    Z2 = A1.dot(W2.T) + b2
    A2 = relu(Z2)

    # output layer
    Z3 = A2.dot(W3.T) + b3
    A3 = softmax(Z3)
    return Z1, A1, Z2, A2, Z3, A3

def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(
        Y.size), Y] = 1  # go through each row and then set the position specified by the label in Y at each column to be 1
    return one_hot_Y

def deriv_relu(Z):
    return Z > 0

def back_prop(Z1, Z2, A1, A2, A3, W2, W3, Y, X):
    n = Y.size
    one_hot_Y = one_hot(Y)

    # output layer
    dZ3 = A3 - one_hot_Y
    dW3 = 1/n * dZ3.T.dot(A2)
    db3 = 1/n * np.sum(dZ3, axis=0, keepdims=True)

    # 2nd hidden layer
    dA2 = dZ3.dot(W3)
    dZ2 = dA2 * deriv_relu(Z2)
    dW2 = (1 / n) * dZ2.T.dot(A1)
    db2 = (1 / n) * np.sum(dZ2, axis=0, keepdims=True)

    # 1st hidden layer
    dA1 = dZ2.dot(W2)
    dZ1 = dA1 * deriv_relu(Z1)
    dW1 = (1 / n) * dZ1.T.dot(X)
    db1 = (1 / n) * np.sum(dZ1, axis=0, keepdims=True)

    return dW1, db1, dW2, db2, dW3, db3

def update_params(dW1, db1, dW2, db2, dW3, db3, W1, b1, W2, b2, W3, b3, alpha):
    W1 = W1 - alpha * dW1
    W2 = W2 - alpha * dW2
    W3 = W3 - alpha * dW3
    b1 = b1 - alpha * db1
    b2 = b2 - alpha * db2
    b3 = b3 - alpha * db3
    return W1, b1, W2, b2, W3, b3


# gradient descent
def get_predictions(A3):
    return np.argmax(A3, axis=1)


def get_accuracy(predictions, Y):
    print(predictions, Y)
    return np.sum(predictions == Y) / Y.size

def gradient_descent(X, Y, alpha, iterations):
    W1, b1, W2, b2, W3, b3 = init_params()
    for i in range(iterations):
        Z1, A1, Z2, A2, Z3, A3 = forward_prop(W1, b1, W2, b2, W3, b3, X)
        dW1, db1, dW2, db2, dW3, db3 = back_prop(Z1, Z2, A1, A2, A3, W2, W3, Y, X)
        W1, b1, W2, b2, W3, b3 = update_params(dW1, db1, dW2, db2, dW3, db3, W1, b1, W2, b2, W3, b3, alpha)
        if i % 10 == 0:
            print("iteration ", i)
            print("accuracy: ", get_accuracy(get_predictions(A3), Y))

    return W1, b1, W2, b2, W3, b3


W1, b1, W2, b2, W3, b3 = gradient_descent(X_train, Y_train, 0.1, 1000)
