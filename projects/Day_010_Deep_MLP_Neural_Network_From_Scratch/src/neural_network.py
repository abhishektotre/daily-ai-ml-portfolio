import numpy as np
from .activations import relu, relu_derivative, softmax, cross_entropy_loss
from .optimizer import AdamOptimizer

class DeepNeuralNetwork:
    def __init__(self, layer_dims, lr=0.01):
        self.layer_dims = layer_dims
        self.lr = lr
        self.weights = []
        self.biases = []
        
        # He (Kaiming) Normal Initialization
        np.random.seed(42)
        for i in range(len(layer_dims) - 1):
            w = np.random.randn(layer_dims[i], layer_dims[i+1]) * np.sqrt(2.0 / layer_dims[i])
            b = np.zeros((1, layer_dims[i+1]))
            self.weights.append(w)
            self.biases.append(b)
            
        self.optimizer = AdamOptimizer(layer_dims, lr=lr)
        
    def forward(self, X):
        self.a_cache = [X]
        self.z_cache = []
        
        curr_a = X
        # Hidden layers
        for i in range(len(self.weights) - 1):
            z = np.dot(curr_a, self.weights[i]) + self.biases[i]
            curr_a = relu(z)
            self.z_cache.append(z)
            self.a_cache.append(curr_a)
            
        # Output layer (Softmax)
        z_out = np.dot(curr_a, self.weights[-1]) + self.biases[-1]
        probs = softmax(z_out)
        self.z_cache.append(z_out)
        self.a_cache.append(probs)
        return probs
        
    def backward(self, y_one_hot):
        m = y_one_hot.shape[0]
        grads_w = []
        grads_b = []
        
        # Output layer gradient
        dz = (self.a_cache[-1] - y_one_hot) / m
        dw = np.dot(self.a_cache[-2].T, dz)
        db = np.sum(dz, axis=0, keepdims=True)
        grads_w.append(dw)
        grads_b.append(db)
        
        # Hidden layers
        for l in range(len(self.weights) - 2, -1, -1):
            da = np.dot(dz, self.weights[l+1].T)
            dz = da * relu_derivative(self.z_cache[l])
            dw = np.dot(self.a_cache[l].T, dz)
            db = np.sum(dz, axis=0, keepdims=True)
            grads_w.insert(0, dw)
            grads_b.insert(0, db)
            
        self.optimizer.step(self.weights, self.biases, grads_w, grads_b)
        
    def fit(self, X, y, epochs=300):
        # Convert y to one-hot
        num_classes = self.layer_dims[-1]
        y_one_hot = np.eye(num_classes)[y]
        
        history = []
        for epoch in range(1, epochs + 1):
            probs = self.forward(X)
            loss = cross_entropy_loss(probs, y_one_hot)
            self.backward(y_one_hot)
            
            preds = np.argmax(probs, axis=1)
            acc = np.mean(preds == y)
            history.append({"epoch": epoch, "loss": float(loss), "accuracy": float(acc)})
        return history
