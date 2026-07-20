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

        if gin_channels != 0:
            self.cond_layer = nn.Conv1d(
                gin_channels,
                2 * hidden_channels * n_layers,
                1                
            )

            self.cond_layer = nn.utils.weight_norm(self.cond_layer)



    def forward(self, x, x_mask, g=None):
        output = torch.zeros_like(x)

        if g is not None:
            g = self.cond_layer(g)
           
        for i in range(self.n_layers):
            x_in = self.in_layers[i](x)

            if g is not None:
                cond_offset = i * 2 * self.hidden_channels

                g_l = g[:, cond_offset:cond_offset + 2 * self.hidden_channels, : ]

                x_in = x_in + g_l


            acts = torch.tanh(
                x_in[:, :self.hidden_channels, :],
            ) * torch.sigmoid(
                x_in[:, self.hidden_channels:, :],
            )

            acts = self.drop(acts)

            res_skip_acts = self.res_skip_layers[i](acts)

            if i < self.n_layers - 1:
                res_acts = res_skip_acts[:, :self.hidden_channels, :]
                skip_acts = res_skip_acts[:, self.hidden_channels:, :]

                x  = (x + res_acts) * x_mask

                output = output + skip_acts
            else:
                output = res_skip_acts + output

        return output * x_mask
        
class ResidualCouplingLayer(nn.Module):

    def __init__(self,
                channels,
                hidden_channels,
                kernel_size,
                dilation_rate,
                n_layers,
                gin_channels=0,
                p_dropout=0.0,
                mean_only=False
            ):
        super().__init__()
        
        self.channels = channels
        self.hidden_channels = hidden_channels
        self.half_channels = channels // 2
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.gin_channels = gin_channels
        self.mean_only = mean_only

        #self.drop = nn.Dropout(p_dropout)

        self.pre = nn.Conv1d(
            self.half_channels,
            hidden_channels,
            1
        )

        self.enc = WN(
            hidden_channels,
            kernel_size,
            dilation_rate,
            n_layers,
            gin_channels=gin_channels,
            p_dropout=p_dropout
        )

        if mean_only:
            self.post = nn.Conv1d(
                hidden_channels,
                self.half_channels,
                1
            )
        else:
            self.post = nn.Conv1d(
                hidden_channels,
                self.half_channels * 2,
                1
            )

        self.post.weight.data.zero_()
        self.post.bias.data.zero_()
 
    def forward(self, x, x_mask, g=None, reverse=False):
        x0, x1 =torch.split(x, [self.half_channels, self.half_channels], 1)
        h = self.pre(x0) * x_mask
        h = self.enc(h, x_mask, g=g)
        stats = self.post(h) * x_mask


        if self.mean_only:
            m = stats
            logs = torch.zeros_like(m)

        else:
            m, logs = torch.split(stats, [self.half_channels, self.half_channels], 1)


        if not reverse:
            x1 = m + x1 * torch.exp(logs)
            x1 *= x_mask
        else:
            x1 = (x1 - m) * torch.exp(-logs)
            x1 *= x_mask
            
        x =torch.cat([x0, x1], dim=1)

        return x


class Flip(nn.Module):

    def forward(self, x, *args, reverse=False, **kwargs):
        
        x = torch.flip(x, [1])

        return x
    

    
        