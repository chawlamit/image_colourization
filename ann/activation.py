"""
Activation functions for neural network layers
"""
from sympy import Symbol, lambdify, exp


def definition(symbols, func_def):
    """
    Helper functions to generate function from sympy function definition
    """
    return lambdify(symbols, func_def)


def derivative(symbols, func_def):
    """
    Helper functions to generate function derivative from sympy function definition
    """
    return lambdify(symbols, func_def.diff())


class Sigmoid:
    x = Symbol('x')
    func_def = 1 / (1 + exp(-x))

    eval = definition(x, func_def)
    prime = derivative(x, func_def) # sigma * (1 - sigma)


class Linear:
    x = Symbol('x')
    func_def = x

    eval = definition(x, func_def)
    prime = derivative(x, func_def)


if __name__ == '__main__':
    sigmoid = Sigmoid
    print(sigmoid.eval(-20))
