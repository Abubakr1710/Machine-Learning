import numpy as np

class LinearRegression:

    def __init__(self, lr = 0.001, n_iters = 1000):
        self.lr = lr # learning rate
        self.n_iters = n_iters # number of iterations
        self.weights = None # weights
        self.bias = None # bias

    def fit(self, X, y):
        n_samples, n_features = X.shape # number of samples and features
        self.weights = np.zeros(n_features) # initialize weights
        self.bias = 0 # initialize bias

        for _ in range(self.n_iters):

            y_pred = np.dot(X, self.weights) + self.bias # prediction

            dw = (1/n_samples)*np.dot(X.T, (y_pred - y)) # gradient of weights
            db = (1/n_samples)*np.sum(y_pred - y) # gradient of bias

            self.weights = self.weights - self.lr*dw # update weights
            self.bias = self.bias - self.lr*db # update bias

    def predict(self, X):
        y_pred = np.dot(X, self.weights) + self.bias # prediction
        return y_pred
