import numpy as np

def relu(z):
    return np.maximum(0, z)

def relu_derivative(z):
    return (z > 0).astype(float)

def softmax(z):
    # Numerically stable softmax
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

def cross_entropy_loss(probs, y_one_hot):
    eps = 1e-15
    probs = np.clip(probs, eps, 1 - eps)
    return -np.mean(np.sum(y_one_hot * np.log(probs), axis=1))
