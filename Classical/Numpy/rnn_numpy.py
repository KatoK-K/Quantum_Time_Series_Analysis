"""
numpyでRNNの実装をします
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


class Activation:

    def __init__(self):
        self.type = 'Softmax'
        self.eps = 1e-15

    def forward(self, Z):

        self.A = np.tanh(Z)

        return self.A


class Tanh:

    def __init__(self):
        self.type = 'Tanh'

    def forward(self, Z):

        self.A = np.tanh(Z)

        return self.A

    def backward(self, dA):

        dZ = dA * (1 - np.power(self.A, 2))

        return dZ


class SSE:

    def __init__(self):
        self.type = 'MSE'
        self.eps = 1e-15

    def forward(self, Y_hat, Y):

        self.Y = Y
        self.Y_hat = Y_hat

        _loss = (self.Y - self.Y_hat) ** 2

        loss = np.sum(_loss, axis=0) / 2

        return np.squeeze(loss)

    def backward(self):

        grad = self.Y_hat - self.Y

        return grad


class SGD:

    def __init__(self, lr=0.0075, beta=0.9):
        """
        Parameters
        ----------
        lr : int, default: 0.0075
            Learing rate to use for the gradient descent.
        beta : int, default: 0.9
            Beta parameter.
        """
        self.beta = beta
        self.lr = lr

    def optim(self, weights, gradients, velocities=None):
        """
        Parameters
        ---------
        weights : numpy.array
            Weigths of a given layer.
        bias : numpy.array
            Bias of a given layer.
        dW : numpy.array
            The gradients of the weights.
        db : numpy.array
            The gradients of the bias
        velocities : tuple
            Tuple containing the velocities to compute the gradient
            descent with momentum.
        Returns
        -------
        weights : numpy.array
            Updated weigths of the given layer.
        bias : numpy.array
            Updated bias of the given layer.
        (V_dW, V_db) : tuple
            Tuple of ints containing the velocities for the weights
            and biases.
        """
        if velocities is None: velocities = [0 for weight in weights]

        velocities = self._update_velocities(
            gradients, self.beta, velocities
        )
        new_weights = []

        for weight, velocity in zip(weights, velocities):
            weight -= self.lr * velocity
            new_weights.append(weight)

        return new_weights, velocities

    def _update_velocities(self, gradients, beta, velocities):
        """
        Updates the velocities of the derivates of the weights and
        bias.
        """
        new_velocities = []

        for gradient, velocity in zip(gradients, velocities):
            new_velocity = beta * velocity + (1 - beta) * gradient
            new_velocities.append(new_velocity)

        return new_velocities


class RNN:

    def __init__(self, input_dim, output_dim, hidden_dim):

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim

        params = self._initialize_params(input_dim, output_dim, hidden_dim)

        self.V = params[0]
        self.W = params[1]
        self.U = params[2]
        self.c = params[3]
        self.b = params[4]

        self.activation = Activation()
        self.oparams = None

    def foward(self, x):
        self.x = x

        self.layers_tanh = [Tanh() for _ in x]
        hidden = np.zeros((self.hidden_dim, 1))

        self.hidden_list = [hidden]
        self.y_preds = []

        for input_x, layer_tanh in zip(x, self.layers_tanh):
            input_tanh = np.dot(self.W, input_x) + np.dot(self.U, hidden) + self.b
            hidden = layer_tanh.forward(input_tanh)
            self.hidden_list.append(hidden)

        input_activation = np.dot(self.V, hidden) + self.c
        y_pred = self.activation.forward(input_activation)
        self.y_preds.append(y_pred)

        return self.y_preds

    def loss(self, Y):
        """
        SSE
        :param Y:
        :return:
        """
        self.Y = Y
        self.layers_loss = [SSE() for _ in Y]
        cost = 0

        for y_pred, y, layer in zip(self.y_preds, self.Y, self.layers_loss):
            cost += layer.forward(y_pred, y)

        return cost

    def backward(self):
        """
        Computes the backward propagation of the model
        :return:
        """
        gradients = self._define_gradients()
        self.dW, self.dU, self.dV, self.db, self.dc, dhidden_next = gradients

        for index, layer_loss in reversed(list(enumerate(self.layers_loss))):

            dy = layer_loss.backward()

            # hidden actual
            hidden = self.hidden_list[index + 1]
            hidden_prev = self.hidden_list[index]

            # gradients y
            self.dV += np.dot(dy, hidden.T)
            self.dc += dy
            dhidden = np.dot(self.V.T, dy) + dhidden_next

            # gradients a
            dtanh = self.layers_tanh[index].backward(dhidden)
            self.db += dtanh
            self.dW += np.dot(dtanh, self.x[index].T)
            self.dU += np.dot(dtanh, hidden_prev.T)
            dhidden_next = np.dot(self.U.T, dtanh)

    def optimize(self, method):
        weights = [self.V, self.W, self.U, self.c, self.b]
        gradients = [self.dV, self.dW, self.dU, self.dc, self.db]

        weights, self.oparams = method.optim(weights, gradients, self.oparams)
        # print("oparams", self.oparams)
        self.V, self.W, self.U, self.c, self.b = weights

    def _initialize_params(self, input_dim, output_dim, hidden_dim):
        den = np.sqrt(hidden_dim)

        weights_V = np.random.randn(output_dim, hidden_dim) / den
        bias_c = np.zeros((output_dim, 1))

        weights_W = np.random.randn(hidden_dim, input_dim) / den
        weights_U = np.random.randn(hidden_dim, hidden_dim) / den
        bias_b = np.zeros((hidden_dim, 1))

        return [weights_V, weights_W, weights_U, bias_c, bias_b]

    def _define_gradients(self):

        dW = np.zeros_like(self.W)
        dU = np.zeros_like(self.U)
        dV = np.zeros_like(self.V)

        db = np.zeros_like(self.b)
        dc = np.zeros_like(self.c)

        da_next = np.zeros_like(self.hidden_list[0])

        return dW, dU, dV, db, dc, da_next


if __name__ == '__main__':

    """
    0. 初期化
    """
    np.random.seed(1)

    """
    1. データの準備
    """

    def sin(x, T=100):
        return np.sin(2.0 * np.pi * x / T)

    def add_noise(T=100, noise_level=0.01):
        x = np.arange(0, 2*T + 1)
        noise = noise_level * np.random.uniform(low=-1.0, high=1.0, size=len(x))
        return sin(x) + noise

    T = 100
    f = add_noise(T).astype(np.float32)

    length = len(f)

    # どれだけ過去を振り返るか
    maxlen = 5

    x = []
    t = []

    for i in range(length - maxlen):
        x.append((f[i:i+maxlen]))
        t.append(f[i+maxlen])

    x = np.array(x).reshape(-1, maxlen, 1)
    t = np.array(t).reshape(-1, 1)

    target_length = len(t)

    # trainデータとtestデータに分割
    x_train, x_val, t_train, t_val = train_test_split(x, t, test_size=0.2, shuffle=False)

    """
    2. モデル構築
    """

    input_dim = 1
    output_dim = 1
    hidden_dim = 50

    model = RNN(input_dim, output_dim, hidden_dim)
    optim = SGD(lr=0.01)
    costs = []

    """
    3. モデルの学習
    """

    epochs = 1000
    batch_size = 100

    n_batches_train = x_train.shape[0] // batch_size + 1
    n_batches_val = x_val.shape[0] // batch_size + 1

    hist = []
    pred_hist = []
    y_hist = []

    for epoch in range(epochs):

        index = epoch % len(x_train)
        X = x_train[index]
        Y = t_train[index]

        X = X.reshape(len(X), 1, -1)

        y_hist.append(Y[0])
        y_preds = model.foward(X)
        pred = y_preds[0][0]
        cost = model.loss(Y)
        model.backward()
        # optimize
        model.optimize(optim)

        print("Cost after iteration %d: %f" % (epoch, cost))

        # if cost < 0.5:
        pred_hist.append(pred)
        hist.append(cost)

    plt.plot([i for i in range(len(hist))], hist)
    # plt.plot([i for i in range(len(hist))], pred_hist)
    plt.show()

    """
    4. モデルの評価
    """

    y_hist = []
    pred_hist = []

    for index in range(len(t_val)):

        X = x_val[index]
        Y = t_val[index]

        X = X.reshape(len(X), 1, -1)

        y_hist.append(Y[0])
        y_preds = model.foward(X)
        pred_hist.append(y_preds[0][0])

    plt.plot([i for i in range(len(y_hist))], y_hist)
    plt.plot([i for i in range(len(y_hist))], pred_hist)
    plt.show()

