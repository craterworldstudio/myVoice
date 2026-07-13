import torch
from torch import nn
from torch.nn import functional as F

class LayerNorm(nn.Module):
    def __init__(self, channels, eps=1e-5):
        super().__init__()

        self.channels = channels
        self.eps = eps

        self.gamma = nn.Parameter(torch.ones(channels))
        self.beta = nn.Parameter(torch.zeros(channels))

    def forward(self, x):
        x = x.transpose(1, 2)

        x = F.layer_norm(
            x,
            (self.channels,),
            self.gamma,
            self.beta,
            self.eps
        )

        return x.transpose(1, 2)
    

class WN(nn.Module):

    def __init__(self,
        hidden_channels,
        kernel_size,
        dilation_rate,
        n_layers,
        gin_channels=0,
        p_dropout=0
        ):
        super().__init__()

        assert kernel_size % 2 == 1

        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.gin_channels = gin_channels

        self.in_layers = nn.ModuleList()
        self.res_skip_layers = nn.ModuleList()

        self.drop = nn.Dropout(p_dropout)

        for i in range(n_layers):
            dilation = dilation_rate ** i

            padding = int(
                (kernel_size * dilation - dilation) / 2
            )

            in_layer = nn.Conv1d(
                hidden_channels,
                2 * hidden_channels,
                kernel_size=kernel_size,
                dilation=dilation,
                padding=padding
            )

            in_layer = nn.utils.weight_norm(in_layer)
            self.in_layers.append(in_layer)

            if i < n_layers - 1:
                res_skip_channels = 2 * hidden_channels
            else:
                res_skip_channels = hidden_channels

            res_skip_layer = nn.Conv1d(
                hidden_channels,
                res_skip_channels,
                1
            )

            res_skip_layer = nn.utils.weight_norm(res_skip_layer)
            self.res_skip_layers.append(res_skip_layer)

            