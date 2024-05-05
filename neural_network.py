from activation_functions import Sigmoid
import numpy as np
import matplotlib.pyplot as plt

class NeuralNetwork:
    def __init__(self, max_epochs=1000, hidden_layers=[], weights = [], out_neurons_num=1, learning_rate = 0.01):
        self.max_epochs = max_epochs
        self.layers = hidden_layers
        self.weights = weights
        self.network = []
        self.delta = []
        self.out_num = out_neurons_num
        self.learning_rate = learning_rate
        self.train_loss = []
        self.test_loss = []
        self.xsize = 0
        self.ysize = 0
        self.hidden_layers_sizes = []
        if self.weights != []:
            print(self.weights)
            self.get_structure()

    def get_structure(self):
        self.xsize = len(self.weights[0][0])
        self.ysize = len(self.weights[-1])
        for w in self.weights[:-1]:
            self.hidden_layers_sizes.append(len(w))
        print(f"input size {self.xsize}, hidden layers {self.hidden_layers_sizes}, output size {self.ysize}")
        if self.hidden_layers_sizes != self.layers: raise Exception("Structure Inconsistency")
         
    def create_network(self, x, y):
        if self.weights == []:
            k = len(x)
            X = x
            for idx, ni in enumerate(self.layers):
                self.weights.append(2*np.random.random((ni, k)) - 1)
                x = Sigmoid(np.matmul(x, self.weights[idx].T))
                self.network.append(x)
                k = ni
            self.weights.append(2*np.random.random((self.out_num, k)) - 1)
            x = Sigmoid(np.matmul(x, self.weights[-1].T))
            self.network.append(x)
            e = y - x
            delta = x * (1 - x) * e
            self.delta.insert(0, delta)
            net = [X] + self.network
            for i in np.arange(len(self.weights) - 1, 0, -1):
                x = net[i]
                e = np.matmul(self.weights[i].T, delta)
                delta = x * (1 - x) * e
                self.delta.insert(0, delta)
        else:
            k = len(x)
            X = x
            for idx, ni in enumerate(self.layers):
                x = Sigmoid(np.matmul(x, self.weights[idx].T))
                self.network.append(x)
                k = ni
            x = Sigmoid(np.matmul(x, self.weights[-1].T))
            self.network.append(x)
            e = y - x
            delta = x * (1 - x) * e
            self.delta.insert(0, delta)
            net = [X] + self.network
            for i in np.arange(len(self.weights) - 1, 0, -1):
                x = net[i]
                e = np.matmul(self.weights[i].T, delta)
                delta = x * (1 - x) * e
                self.delta.insert(0, delta)
        
    def update_weights(self, x):
        net = [x] + self.network
        layer = self.layers + [self.out_num]
        for i in np.arange(len(self.delta)):
            self.weights[i] = self.weights[i] + ((self.learning_rate * self.delta[i]).reshape(layer[i], 1) * net[i])

    def update_neurons(self, x, y):
        for idx, ni in enumerate(self.layers):
            x = Sigmoid(np.matmul(x, self.weights[idx].T))
            self.network[idx] = x
        x = Sigmoid(np.matmul(x, self.weights[-1].T))
        self.network[-1] = x
        e = y - x
        delta = x * (1 - x) * e
        self.delta[-1] = delta
    
    def update_delta(self):
        net = [[]] + self.network
        delta = self.delta[-1]
        for i in np.arange(len(self.weights) - 1, 0, -1):
            x = net[i]
            e = np.matmul(self.weights[i].T, delta)
            delta = x * (1 - x) * e
            self.delta[i - 1] = delta

    def calculate_loss(self, X, y):
        n = len(X)
        ce = 0
        for xi, yi in zip(X, y):
            y_pred = self.predict_proba(xi)
            ce += (yi * np.log(y_pred) + (1 - yi) * np.log(1 - y_pred))
        return -ce / n

    def fit(self, X, y, X_test=None, y_test=None):
        N = len(X)
        self.create_network(X[0], y[0])
        self.update_delta()
        self.update_weights(X[0])
        loss = 0
        loss_t = 0
        ep = 0
        err = 1
        print("epoch 0")
        while err > 0.0001 and ep < self.max_epochs: #and loss >= loss_t:
            ce = 0
            for xi, yi in zip(X, y):
                self.update_neurons(xi,yi)
                self.update_delta()
                self.update_weights(xi)
                #ce += (yi * np.log(self.network[-1]) + (1 - yi) * np.log(1 - self.network[-1]))
            #loss = -ce / N
            loss = self.calculate_loss(X, y)
            ep += 1
            if X_test is not None and y_test is not None:
                loss_t = self.calculate_loss(X_test, y_test)
                self.test_loss.append(loss_t)
                print(f"epoch {ep}; train loss {loss}; test_loss {loss_t}")
            else:    
                print(f"epoch {ep}; train loss {loss}")
            err = loss
            self.train_loss.append(err)
        
    def predict_proba(self, X):
        x = X
        for idx, ni in enumerate(self.layers):
            x = Sigmoid(np.matmul(x, self.weights[idx].T))
            self.network[idx] = x
        y_pred = Sigmoid(np.matmul(x, self.weights[-1].T))
        return y_pred
    
    def predict(self, X, decision_bound=0.5):
        return [(1 if y_pred >= decision_bound else 0) for y_pred in self.predict_proba(X)]
      

