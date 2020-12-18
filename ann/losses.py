from abc import ABC
import numpy as np
from sympy import Symbol, lambdify, log


class Loss(ABC):
    def __init__(self):
        self.eval = None
        self.prime = None

    def loss(self, out_hat, out):
        if out_hat.shape != out.shape:
            raise Exception(f"y_hat({out_hat.shape}) and y({out.shape}) have different shapes")
        return np.mean(self.eval(out_hat, out), axis=0).squeeze()

    def grad(self, out_hat, out):
        if out_hat.shape != out.shape:
            raise Exception(f"y_hat({out_hat.shape}) and y({out.shape}) have different shapes")
        return np.mean(self.prime(out_hat, out), axis=0).squeeze()


class MSE(Loss):
    def __init__(self):
        super().__init__()
        a = Symbol('a')
        b = Symbol('b')
        func_def = ((a - b) ** 2)
        func_diff = 2 * (a - b)

        self.eval = lambdify((a, b), func_def)
        self.prime = lambdify((a, b), func_diff)


class BCE(Loss):
    def __init__(self):
        super().__init__()
        a = Symbol('a')
        b = Symbol('b')
        func_def = -b * log(a) - (1 - b) * log(1 - a)
        func_diff = a - b

        self.eval = lambdify((a, b), func_def)
        self.prime = lambdify((a, b), func_diff)


class CCE(Loss):
    def __init__(self):
        super(CCE, self).__init__()

    def loss(self, out_hat, out):
        return np.mean(-np.sum(np.multiply(out, np.log(out_hat)), axis=1))

    def grad(self, out_hat, out):
        return np.mean(out_hat)


if __name__ == "__main__":
    import numpy as np
    y_hat = np.array([1, 3, 5])
    y = np.array([1, 5, 9])
    mse = MSE()
    print(mse.loss(y_hat, y), mse.grad(y_hat, y))
