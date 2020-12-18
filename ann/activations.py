"""
Activation functions for neural network layers
"""
from sympy import Symbol, lambdify, exp
import numpy as np

from ann.layer import Layer


class Sigmoid(Layer):

    def __init__(self):
        self._input = None
        self.grad = None

        # function definition
        x = Symbol('x')
        definition = 1 / (1 + exp(-x))
        self.eval = lambdify(x, definition)
        self.prime = lambdify(x, definition * (1 - definition))

    def forward(self, _input):
        self._input = _input
        return self.eval(_input)

    def back_prop(self, dl_dy):
        gradient = self.prime(self._input)
        out = np.mean(np.multiply(gradient, dl_dy), axis=0)
        return out

    def __str__(self):
        return str(__class__)


class ReLU(Layer):
    def __init__(self):
        self._input = None
        self.grad = None

    def forward(self, _input):
        self._input = _input
        return np.where(_input >= 0, _input, 0)

    def back_prop(self, dl_dy):
        return np.mean(np.multiply(np.where(self._input > 0, 1, 0), dl_dy), axis=0)

    def __str__(self):
        return str(__class__)


class Softmax(Layer):
    def __init__(self):
        self._input = None
        self.grad = None

    def forward(self, _input):
        self._input = _input
        term = np.exp(_input - np.max(_input, axis=1)[:, None])
        return term/term.sum(axis=1)[:, None]

    def back_prop(self, dl_dy):
        fwd = self.forward(self._input)
        gradient = np.multiply(fwd, (1 - fwd))
        out = np.mean(np.multiply(gradient, dl_dy), axis=0)
        return out


if __name__ == '__main__':
    sigmoid = Sigmoid()
    inp = np.arange(4).reshape(1, -1)
    print(sigmoid.forward(inp))
    # print(sigmoid.back_prop(0.5))
    print(sigmoid.back_prop(inp / 10))

    relu = ReLU()
    print(relu.forward(inp))
    print(relu.back_prop(0.5))

    soft = Softmax()
    print(soft.forward(inp))
    print(soft.back_prop(np.array([0.1, 0.2, 0.3, 0.4])))
