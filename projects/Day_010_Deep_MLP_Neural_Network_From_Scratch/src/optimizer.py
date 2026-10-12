import numpy as np

class AdamOptimizer:
    def __init__(self, layers_dims, lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m_w = [np.zeros((layers_dims[i], layers_dims[i+1])) for i in range(len(layers_dims)-1)]
        self.v_w = [np.zeros((layers_dims[i], layers_dims[i+1])) for i in range(len(layers_dims)-1)]
        self.m_b = [np.zeros((1, layers_dims[i+1])) for i in range(len(layers_dims)-1)]
        self.v_b = [np.zeros((1, layers_dims[i+1])) for i in range(len(layers_dims)-1)]
        self.t = 0
        
    def step(self, weights, biases, grads_w, grads_b):
        self.t += 1
        for i in range(len(weights)):
            # Update moments for weights
            self.m_w[i] = self.beta1 * self.m_w[i] + (1 - self.beta1) * grads_w[i]
            self.v_w[i] = self.beta2 * self.v_w[i] + (1 - self.beta2) * (grads_w[i] ** 2)
            m_hat_w = self.m_w[i] / (1 - self.beta1 ** self.t)
            v_hat_w = self.v_w[i] / (1 - self.beta2 ** self.t)
            weights[i] -= self.lr * m_hat_w / (np.sqrt(v_hat_w) + self.eps)
            
            # Update moments for biases
            self.m_b[i] = self.beta1 * self.m_b[i] + (1 - self.beta1) * grads_b[i]
            self.v_b[i] = self.beta2 * self.v_b[i] + (1 - self.beta2) * (grads_b[i] ** 2)
            m_hat_b = self.m_b[i] / (1 - self.beta1 ** self.t)
            v_hat_b = self.v_b[i] / (1 - self.beta2 ** self.t)
            biases[i] -= self.lr * m_hat_b / (np.sqrt(v_hat_b) + self.eps)
