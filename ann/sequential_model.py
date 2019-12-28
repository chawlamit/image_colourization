"""
Dense Sequential Neural Network Model
"""
from ann.hidden_layer import HiddenLayer
from ann.activation import Linear
from ann.loss_functions import MSE
import numpy as np
from tqdm import tqdm


class Sequential:
    def __init__(self):
        self.layers = []

    def add(self, layer: HiddenLayer):
        layer.initialize()
        self.layers.append(layer)

    def fit(self, x_train, y_train, epoch=10, loss=MSE, batch_size=10, rate=0.001):
        """
        Train the neural network model
        :param epoch:
        :param rate:
        :param x_train: (n, input_features)
        :param y_train: (last_layer_size, 1)
        :param loss: loss function used to train the model
        :param batch_size: No. of data points used in for batch gradient descent
        :return: None
        """
        # TODO - learn bias
        with tqdm(total=epoch * len(x_train)) as pbar:
            for ep in range(epoch):
                for pt_i in range(len(x_train)):
                    data_pt = x_train[pt_i]
                    _out = data_pt
                    out_layer_wise = [_out]
                    for hl in self.layers:
                        _out = hl.out(_out)
                        out_layer_wise.append(_out)
                    # dL/dout vector at output layers
                    loss_prime_t = loss.prime(_out, y_train[pt_i])
                    # Let's modify the wirghts and back-prop the error
                    for t in range(len(self.layers)-1, -1, -1):
                        hl = self.layers[t]
                        # i, j - node i in layer t-1 to j in t
                        sigma_prime = self.layers[t].prime(out_layer_wise[t])
                        self.layers[t].W = self.layers[t].W - rate * np.matmul(
                            np.reshape((loss_prime_t * sigma_prime), (self.layers[t].units, 1)),
                            np.reshape(out_layer_wise[t], (1, hl.prev_layer_dim)))

                        loss_prime_t = np.matmul(
                            (loss_prime_t * self.layers[t].prime(out_layer_wise[t])),
                            self.layers[t].W)

                    pbar.update()
                    pbar.set_description(f"epoch: {ep+1}, pt: {pt_i}, Avg loss: {np.sum(loss.eval(_out, y_train[pt_i]))}")

    def predict(self, x_test, y_test=None, loss=MSE):
        out = []
        error = []
        for pt_i, data_pt in enumerate(x_test):
            _out = data_pt
            for hl in self.layers:
                _out = hl.out(_out)
            if y_test:
                _error = np.sum(loss.eval(_out, y_test[pt_i]))
            out.append(_out)
            if y_test:
                error.append(_error)
        if y_test:
            print("Average Loss: ", np.mean(error))
        return out


if __name__ == '__main__':
    model = Sequential()
    model.add(HiddenLayer(units=18, prev_layer_dim=9))
    model.add(HiddenLayer(units=9, prev_layer_dim=18))
    model.add(HiddenLayer(units=3, prev_layer_dim=9, activation=Linear, is_output=True))

    x_data = [np.array([i for i in range(9)])]
    y_data = [np.array([1, 2, 3])]
    model.fit(x_data, y_data)

    print("out", model.predict(x_data, y_data))
