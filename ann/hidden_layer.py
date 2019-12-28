from abc import ABC, abstractmethod
from ann.activation import Sigmoid
import numpy as np


class HiddenLayer:
    """
    Base class for hidden Layers of the network
    """
    # TODO - use bias value as well
    def __init__(self, units: int, prev_layer_dim: int, activation=Sigmoid, is_output=False):
        """
        Initialize a hidden layer for the ann
        :param units: No of nodes in the hidden layer
        :param activation: Activation function for each node - defaults to sigmoid
        """
        self.activation = activation
        self.is_output = is_output

        self.units = units
        self.prev_layer_dim = prev_layer_dim
        self.dim = (units, prev_layer_dim)

        self.bias = None
        self.W = None

    def initialize(self):
        """
        Initialize weight vectors Normal Distribution with mu=0 and sd=0.01
        :return:
        """
        self.bias = np.random.normal(0, 0.01)
        self.W = np.random.normal(loc=0, scale=0.01, size=self.dim)

    def out(self, _input: np.array):
        w_dot_x = np.matmul(self.W, _input)
        # if not self.is_output:
        #     w_dot_x = np.append(self.bias, w_dot_x)
        return self.activation.eval(w_dot_x)

    def prime(self, _input: np.array, unit=None):
        """
        evaluates a'(W.in)
        Also capable of calculating loss' for the single given unit index
        """
        if unit is None:
            return self.activation.prime(np.matmul(self.W, _input))
        else:
            return self.activation.prime(np.matmul(self.W[unit, ], _input))

    # def back_prop(self):
    #     pass


if __name__ == '__main__':
    inp = np.random.randint(0, 255, 400)
    l1 = HiddenLayer(units=10, prev_layer_dim=400)
    l1.initialize()
    print(l1.out(inp))
    print(l1.bias)