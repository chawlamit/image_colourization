"""
Activation functions for neural network layers
"""
from sympy import Symbol, lambdify, exp


class Activation:
    """
    base class for activation functions
    Used as as argument for hidden layer
    """
    symbols = None
    func_def = None

    @classmethod
    def func(cls):
        return lambdify(cls.symbols, cls.func_def)

    @classmethod
    def prime(cls):
        return lambdify(cls.symbols, cls.func_def.diff())


class Sigmoid(Activation):
    x = Symbol('x')
    symbols = (x, )
    func_def = 1 / (1 + exp(-x))


if __name__ == '__main__':
    sigmoid = Sigmoid.func()
    sigmoid_prime = Sigmoid.prime()
    print(sigmoid(-20))
