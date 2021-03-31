import numpy as np
import torch
import torch.nn as nn

TH_DTYPE = torch.float32

MOMENTUM = 0.99
EPSILON = 1e-6
DELTA_CLIP = 50.0


class FCSubNet(torch.nn.Module):

    def __init__(self, config):
        super().__init__()
        self.dim = config.dim
        layer_dims = [self.dim] + config.num_hiddens  # layer_dims: [2, 32, 128, 32]
        self.bn_layers = torch.nn.ModuleList([
            torch.nn.BatchNorm1d(self.dim, eps=1e-6, momentum=0.99)])
        self.dense_layers = torch.nn.ModuleList([])
        for i in range(len(layer_dims) - 1):
            self.dense_layers.append(torch.nn.Linear(
                layer_dims[i], layer_dims[i + 1], bias=False))
            self.bn_layers.append(torch.nn.BatchNorm1d(
                layer_dims[i + 1], eps=1e-6, momentum=0.99))

        # output layers
        self.dense_layers.append(torch.nn.Linear(
            layer_dims[-1], self.dim, bias=True))
        self.bn_layers.append(torch.nn.BatchNorm1d(
            self.dim, eps=1e-6, momentum=0.99))

        # initializing batchnorm layers
        for layer in self.bn_layers:
            torch.nn.init.uniform_(layer.weight, 0.1, 0.5)
            torch.nn.init.normal_(layer.bias, 0.0, 0.1)

        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.bn_layers[0](x)
        for i in range(len(self.dense_layers) - 1):
            x = self.dense_layers[i](x)
            x = self.bn_layers[i + 1](x)
            x = self.relu(x)
        x = self.dense_layers[-1](x)
        x = self.bn_layers[-1](x) / self.dim
        return x


class FCLSTMSubNet(torch.nn.Module):

    def __init__(self, config):
        super().__init__()
        self.dim = config.dim
        layer_dims = [self.dim, config.lstm_hidden_size] + config.num_hiddens
        self.bn_layers = torch.nn.ModuleList([])
        self._layers = torch.nn.ModuleList([])
        self._layers.append(
            torch.nn.LSTM(input_size=self.dim, hidden_size=config.lstm_hidden_size, num_layers=config.lstm_num_layers))
        for i in range(1, len(layer_dims) - 1):
            self._layers.append(torch.nn.Linear(
                layer_dims[i], layer_dims[i + 1], bias=False))
        for i in range(len(layer_dims)):
            self.bn_layers.append(torch.nn.BatchNorm1d(
                layer_dims[i], eps=1e-6, momentum=0.99))
        # output layers
        self._layers.append(torch.nn.Linear(
            layer_dims[-1], self.dim, bias=True))
        self.bn_layers.append(torch.nn.BatchNorm1d(
            self.dim, eps=1e-6, momentum=0.99))

        # initializing batchnorm layers
        for layer in self.bn_layers:
            torch.nn.init.uniform_(layer.weight, 0.1, 0.5)
            torch.nn.init.normal_(layer.bias, 0.0, 0.1)

        self.relu = torch.nn.ReLU()

    def forward(self, x, hidden):
        x = self.bn_layers[0](x)
        x = x.unsqueeze(0)
        x, hidden_ = self._layers[0](x, hidden)
        x = x.squeeze(0)
        x = self.bn_layers[1](x)
        for i in range(len(self._layers) - 2):
            x = self._layers[i + 1](x)
            x = self.bn_layers[i + 2](x)
            x = self.relu(x)
        x = self._layers[-1](x)
        x = self.bn_layers[-1](x) / self.dim
        return x / self.dim, hidden_


def my_sig(x):
    return 2.0 / (1 + torch.exp(-x)) - 1


class FeedForwardModel(nn.Module):

    def __init__(self, config, fbsde):
        super(FeedForwardModel, self).__init__()
        self._config = config
        self.fbsde = fbsde

        self._dim = fbsde.dim
        self._num_time_interval = fbsde.num_time_interval
        self._total_time = fbsde.total_time

        self.register_parameter('y_init', torch.nn.Parameter(
            torch.rand(1).uniform_(config.y_init_range[0],
                                   config.y_init_range[1])))
        self.register_parameter('z_init', torch.nn.Parameter(
            torch.rand(1, config.dim).uniform_(config.z_init_range[0],
                                               config.z_init_range[1])))
        if config.lstm == True:
            self._subnetworkList = nn.ModuleList([FCLSTMSubNet(config)])
        else:
            self._subnetworkList = nn.ModuleList([FCSubNet(config)])

    def forward(self, dw):
        R = self.fbsde._R
        num_sample = dw.shape[0]
        gamma = self.fbsde.gamma(num_sample)
        sigma = self.fbsde.sigma(num_sample)
        G = self.fbsde.G_th(num_sample)

        all_one_vec = torch.ones((num_sample, 1), dtype=TH_DTYPE)
        y = all_one_vec * self.y_init
        y = y.unsqueeze(2)
        z = all_one_vec * self.z_init
        z = z.unsqueeze(2)

        x_sample = torch.zeros([num_sample, self._dim, 1])
        hidden = (torch.randn(self._config.lstm_num_layers, num_sample, self._config.lstm_hidden_size),
                  torch.randn(self._config.lstm_num_layers, num_sample, self._config.lstm_hidden_size))
        totalx = []
        totalu = []
        time_stamp = np.arange(0, self.fbsde.num_time_interval) * self.fbsde.delta_t
        for t in range(0, self._num_time_interval):
            totalx.append(x_sample)
            u = (-1 / R) * torch.bmm(torch.transpose(gamma, 1, 2), z)
            if self._config.constrained == True:
                u = torch.clamp(u, self._config.u_threhold_min, self._config.u_threhold_max)
            totalu.append(u)
            i1 = self.fbsde.delta_t * self.fbsde.h_th(time_stamp[t], x_sample, y, z, u)
            i2 = self.fbsde.delta_t * torch.bmm(torch.transpose(z, 1, 2), gamma) * u
            dw_ = dw[:, :, t].unsqueeze(2)
            i3 = torch.bmm(torch.transpose(z, 1, 2), dw_)
            y = y - i1 + i2 + i3

            if t == self.fbsde.num_time_interval - 1:
                break

            item1 = (self.fbsde.f_th(x_sample) * self.fbsde.delta_t).unsqueeze(2)
            tmp = item1 + torch.bmm(sigma, gamma) * u * self.fbsde.delta_t + torch.mul(G, dw_)
            # tmp = item1 + torch.bmm(sigma, gamma) * u * self.fbsde.delta_t + torch.bmm(sigma, dw_)
            x_sample = x_sample + tmp
            if self._config.lstm == True:
                z, hidden = self._subnetworkList[0](x_sample.squeeze(2), hidden)
            else:
                z = self._subnetworkList[0](x_sample.squeeze(2)) / self._dim
            z = z.unsqueeze(2)

        ye = self.fbsde.g_th(self._total_time, x_sample, u)
        crit = torch.nn.SmoothL1Loss()
        loss = 2 * self._config.DELTA_CLIP * (crit(y.squeeze(2), ye) + torch.mean(ye[:, 0]) ** 2)
        return loss, self.y_init, y, ye, totalx, totalu
