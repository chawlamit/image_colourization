import numpy as np

from ann.layer import Layer


class Linear(Layer):
    """
    Base class for Linear Layers of the network
    """
    def __init__(self, in_dim: int, out_dim: int):
        """
        Initialize a hidden layer for the ann
        """
        self.in_dim = in_dim + 1 #bias
        self.out_dim = out_dim

        # last forwarded input
        self._input = None
        self.W = np.zeros((self.in_dim, self.out_dim))
        self.grad = np.zeros_like(self.W)

    def initialize(self, mu=0, sd=0.1):
        """
        Initialize weight vectors Normal Distribution with mu=0 and sd=0.01
        :return:
        """
        self.W = np.random.normal(loc=mu, scale=sd, size=(self.in_dim, self.out_dim))

    def forward(self, _input: np.array):
        """
        _input (1000, 3)
        """
        if len(_input.shape) == 1: # 1D array
            _input = _input.reshape(1, -1) # make it a matrix

        bias = np.ones(shape=(_input.shape[0], 1))

        self._input = np.concatenate([_input, bias], axis=1)
        # print(self._input.shape, self.W.shape)
        return np.matmul(self._input, self.W) # 100,3 X 3, 2 = 100, 2

    def back_prop(self, dl_dy, lr): # (2, )
        """
        :param dl_dy: out_dim
        calculates dL/dw and back return dL/dx for back prop
        Also capable of calculating loss' for the single given unit index
        """
        # size = self._input.shape[0] # 100
        # out = self.forward(self._input) # (100, 2)
        #
        # _input = self._input.reshape(-1, 1) # (300, 1)
        # self.grad = np.matmul(_input, dl_dy.reshape(1, -1)) # (1, 2)
        # # print("Grad:", self.grad.shape)
        # self.grad = np.split(self.grad, size)
        # self.grad = np.mean(self.grad, axis=0) # make it tensor of shape (size, in_dim, out_dim)
        # # print(self.grad)
        # return np.dot(self.W, dl_dy).squeeze()

        self.grad = np.dot(self._input[:, :-1].T, dl_dy)
        self.W[:-1, :] -= lr * self.grad
        self.W[-1, :] -= (lr * dl_dy).squeeze()
        return np.dot(dl_dy, self.W[:-1, :].T)

    def update(self, lr):
        """
        :param lr: Learning Rate
        """
        self.W -= lr * self.grad

    def __str__(self):
        return f"{self.__class__}: {(self.in_dim, self.out_dim)}"


if __name__ == '__main__':
    inp = np.random.randint(0, 255, 6)
    l1 = Linear(6, 3)
    l1.initialize()
    print(l1.forward(inp))
    print(l1.back_prop(np.array([0.1, 0.2, 0.3])))
