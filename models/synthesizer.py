import math, torch
from torch import nn

from models.modules import ResidualCouplingLayer, Flip

class ResidualCouplingBlock(nn.Module):
    
    def __init__(self, 
                channels,
                hidden_channels,
                kernel_size,
                dilation_rate,
                n_layers,
                n_flows=4,
                gin_channels=0):
        super().__init__()

        self.flows = nn.ModuleList()

        for _ in range(n_flows):
            self.flows.append(
                ResidualCouplingLayer(
                    channels,
                    hidden_channels,
                    kernel_size,
                    dilation_rate,
                    n_layers,
                    gin_channels=gin_channels,
                    mean_only=True
                )
            )
            self.flows.append(Flip())


    def forward(self, x, x_mask, g=None, reverse=False):

        if not reverse:
            for flow in self.flows:
                x = flow(x, x_mask, g=g, reverse=False)
        else:
            for flow in reversed(self.flows):
                x = flow(x, x_mask, g=g, reverse=True)
        return x