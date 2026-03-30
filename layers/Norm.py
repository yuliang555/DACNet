import torch
import torch.nn as nn

class InstanceNorm(nn.Module):

    def __init__(self, dim, norm):
        super(InstanceNorm, self).__init__()
        self.dim = dim
        self.norm = norm

    def forward(self, x, mode:str):
        if mode == 'norm':
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)

        return x

    def _normalize(self, x):
        self.means = torch.mean(x, dim=self.dim, keepdim=True)
        self.stdev = torch.sqrt(torch.var(x, dim=self.dim, keepdim=True) + 1e-5)

        if self.norm == 1:
            x = x - self.means
        elif self.norm == 2:
            x = (x - self.means) / self.stdev

        return x

    def _denormalize(self, x):

        if self.norm == 1:
            x = x + self.means
        elif self.norm == 2:
            x = x * self.stdev + self.means

        return x