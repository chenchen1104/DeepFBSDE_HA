import numpy as np
import torch


class Equation(object):
    def __init__(self, dim, total_time, delta_t):
        self._dim = dim
        self._total_time = total_time
        self._delta_t = delta_t
        self._num_time_interval = int(self._total_time / delta_t)
        self._sqrt_delta_t = np.sqrt(self._delta_t)
        self._y_init = None

    def sample(self, num_sample):
        """Sample forward SDE."""
        raise NotImplementedError

    def f_th(self, x):
        """Generator function in the PDE."""
        raise NotImplementedError

    def g_th(self, t, x, u):
        """Terminal condition of the PDE."""
        raise NotImplementedError

    @property
    def y_init(self):
        return self._y_init

    @property
    def dim(self):
        return self._dim

    @property
    def num_time_interval(self):
        return self._num_time_interval

    @property
    def total_time(self):
        return self._total_time

    @property
    def delta_t(self):
        return self._delta_t

class Aircraft(Equation):
    def __init__(self, dim, total_time, num_time_interval):
        super(Aircraft, self).__init__(dim, total_time, num_time_interval)
        self._x_init = np.zeros(self._dim)  # 状态量分别为[飞机迎角 alpha, 飞机俯仰角theta, 飞机俯仰角速度q]

        self.b_alpha = 0.073
        self.b_delta_z = -0.0035
        self.a_alpha = 0.7346
        self.a_delta_z = -2.8375
        self.a_q = 3.9779

        self.theta_desired = 10

        d = torch.tensor([1.0, 0.3, 0.1])
        self.sigma_ = torch.diag(d)
        self.G_ = torch.tensor([[-self.b_delta_z], [0], [-self.a_delta_z]])
        self.gamma_ = torch.mm(self.sigma_.inverse(), self.G_)
        self._R = 0.1

    def sample(self, num_sample):
        dw = torch.empty(num_sample, self.dim, self.num_time_interval).normal_(std=self._sqrt_delta_t)
        return dw

    def f_th(self, x):
        f1 = -self.b_alpha * x[:, 0]
        f2 = x[:, 2]
        f3 = -self.a_q * x[:, 2] - self.a_alpha * x[:, 0]
        f = torch.cat((f1, f2, f3), 1)
        return f

    def g_th(self, t, x, u):
        u = u.squeeze(2)
        error = x[:, 1] - self.theta_desired
        g = error ** 2 + 0.1 * (x[:, 2]) ** 2 + 0.01 * u ** 2
        return g

    def h_th(self, t, x, y, z, u):
        gamma = self.gamma(z.shape[0])
        temp = torch.bmm(torch.transpose(z, 1, 2), gamma) * (1 / self._R)
        temp1 = torch.bmm(temp, torch.transpose(gamma, 1, 2))
        q = self.g_th(t, x, u).unsqueeze(2)
        h = q - torch.bmm(temp1, z) / 2
        return h

    def sigma(self, length):
        sigma = torch.zeros([length, self.dim, self.dim])
        for i in range(length):
            sigma[i] = self.sigma_
        return sigma

    def gamma(self, length):
        gamma = torch.zeros([length, self.dim, 1])
        for i in range(length):
            gamma[i] = self.gamma_
        return gamma

    def G_th(self, length):
        G = torch.zeros([length, self.dim, 1])
        for i in range(length):
            G[i] = self.G_
        return G


def get_equation(name, dim, total_time, num_time_interval):
    try:
        return globals()[name](dim, total_time, num_time_interval)
    except KeyError:
        raise KeyError("Equation for the required problem not found.")
