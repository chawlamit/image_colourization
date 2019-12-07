from abc import ABC, abstractmethod
from ann.activation import Activation, Sigmoid
import numpy as np


class HiddenLayer:
    """
    Base class for hidden Layers of the network
    """
    def __init__(self, units: int, prev_layer_dim: int, activation=Sigmoid):
        """
        Initialize a hidden layer for the ann
        :param units: No of nodes in the hidden layer
        :param activation: Activation function for each node - defaults to sigmoid
        """
        self.units = units
        self.activation = activation.func()
        self.dim = (units + 1, prev_layer_dim)
        self.W = None

    def initialize(self):
        """
        Initialize weight vectors Normal Distribution with mu=0 and sd=0.01
        :return:
        """
        self.W = np.random.normal(loc=0, scale=0.01, size=self.dim)

    def out(self, inp: np.array):
        return self.activation(np.matmul(self.W, inp))


if __name__ == '__main__':
    inp = np.random.randint(0, 255, 400)
    l1 = HiddenLayer(units=10, prev_layer_dim=400)
    l1.initialize()
    print(l1.out(inp))