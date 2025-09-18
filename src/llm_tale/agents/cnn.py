import torch
from torch import nn
import torch.nn.functional as F


class CnnEncoder(nn.Module):
    def __init__(self, obs_shape, proj_dim=128):
        super().__init__()
        self.obs_shape = obs_shape
        assert len(obs_shape) == 3
        self.repr_dim = 16 * 25 * 25
        self.aug = RandomShiftsAug(4)
        self.convnet = nn.Sequential(
            nn.Conv2d(obs_shape[0], 16, 3, stride=2),
            nn.LeakyReLU(),
            nn.Conv2d(16, 16, 3, stride=1),
            nn.LeakyReLU(),
            nn.Conv2d(16, 16, 3, stride=1),
            nn.LeakyReLU(),
            nn.Conv2d(16, 16, 3, stride=1),
            nn.LeakyReLU(),
        )
        self.trunk = nn.Sequential(nn.Linear(self.repr_dim, proj_dim), nn.LayerNorm(proj_dim), nn.Tanh())

        self.apply(weight_init)

    def forward(self, obs):
        h = self.convnet(obs)
        h = h.view(h.shape[0], -1)
        h = self.trunk(h)
        return h

    def cnn_forward(self, _input, action=None):
        obs_num = self.obs_shape[0] * self.obs_shape[1] * self.obs_shape[2]
        image = _input[:, :obs_num].reshape(-1, self.obs_shape[0], self.obs_shape[1], self.obs_shape[2])
        image = self.aug(image)
        feat = self.forward(image)
        state = _input[:, obs_num:]
        if action is not None:
            state = torch.cat((state, action), dim=1)
        else:
            state = _input[:, obs_num:]
        state = torch.cat((feat, state), dim=1)
        return state


class RandomShiftsAug(nn.Module):
    def __init__(self, pad):
        super().__init__()
        self.pad = pad

    def forward(self, x):
        n, c, h, w = x.size()
        assert h == w
        padding = tuple([self.pad] * 4)
        x = F.pad(x, padding, "replicate")
        eps = 1.0 / (h + 2 * self.pad)
        arange = torch.linspace(-1.0 + eps, 1.0 - eps, h + 2 * self.pad, device=x.device, dtype=x.dtype)[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)

        shift = torch.randint(0, 2 * self.pad + 1, size=(n, 1, 1, 2), device=x.device, dtype=x.dtype)
        shift *= 2.0 / (h + 2 * self.pad)

        grid = base_grid + shift
        return F.grid_sample(x, grid, padding_mode="zeros", align_corners=False)


def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        gain = nn.init.calculate_gain("leaky_relu")
        nn.init.orthogonal_(m.weight.data, gain)
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)
