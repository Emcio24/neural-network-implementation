import numpy as np
import latexify

@latexify.function
def Sigmoid(x):
    return 1 / (1 + np.exp(-x))

def Softmax(x):
    x  = np.subtract(x, np.max(x)) # prevent overflow
    ex = np.exp(x)
    return ex / np.sum(ex)

def tanh(x):
    return np.tanh(x)

@latexify.function
def ReLU(x):
    return max(0, x)
