from sympy import Symbol, lambdify, log
import numpy as np


class MSE:
    y_hat = Symbol('x')
    y = Symbol('y')
    func_def = 0.5 * ((y_hat - y) ** 2)
    func_diff = y_hat - y

    eval = lambdify((y_hat, y), func_def)
    prime = lambdify((y_hat, y), func_diff)

    @classmethod
    def total_loss(cls, out_hat, out):
        return sum(cls.eval(out_hat, out))


class BinaryCrossEntropy:
    y_hat = Symbol('x')
    y = Symbol('y')
    func_def = -y*log(y_hat) -(1 - y)*log(1 - y_hat)
    func_diff = y_hat - y

    eval = lambdify((y_hat, y), func_def)
    prime = lambdify((y_hat, y), func_diff)


if __name__ == "__main__":
    import numpy as np
    a = np.array([1, 3, 5])
    b = np.array([1, 5, 9])

    print(MSE.eval(a, b))