import time, math, torch
from torch import nn
from torch.nn import functional as F
from models.modules import LayerNorm

class MultiHeadAttention(nn.Module):

    def __init__(self,
                  channels, out_channels,
                  n_heads, p_dropout=0.0, window_size=None
                 ):
        
        #For Alastor - window size = 10
        super().__init__()

        assert channels % n_heads == 0

        self.channels = channels
        self.out_channels = out_channels
        self.n_heads = n_heads
        self.window_size = window_size

        self.k_channels = channels  // n_heads

        self.conv_q = nn.Conv1d(channels, channels, 1)
        self.conv_k = nn.Conv1d(channels, channels, 1)
        self.conv_v = nn.Conv1d(channels, channels, 1)

        self.conv_o = nn.Conv1d(channels, out_channels, 1)

        self.drop = nn.Dropout(p_dropout)


        if window_size is not None:
            self.emb_rel_k = nn.Parameter(
                torch.randn(1, 2*window_size + 1, self.k_channels)
            )

            self.emb_rel_v = nn.Parameter(
                torch.randn(1, 2*window_size + 1, self.k_channels)
            )


    def _relative_position_to_absolute_position(self, x):
        batch, heads, length, _ = x.size()

        x = F.pad(x, [0,1])

        x = x.view(
            batch, heads, length * (2 * length)
        )

        x = F.pad(x, [0, length - 1])

        x = x.view(
            batch,
            heads,
            length + 1,
            2 * length - 1
        )

        return x[:, :, :length, length -1:]

    def _absolute_position_to_relative_position(self, x):
        
        batch, heads, length, _ = x.size()

        x = F.pad(x, [0, length - 1])

        x = x.view(
            batch, heads, length * length + length * (length - 1)
        )

        x = F.pad(x, [0, length])

        x = x.view(
            batch, heads, length, 2 * length
        )

        return x[:, :, :, 1:]
    
    def _get_relative_embeddings(self, relative_embeddings, length):

        pad_length = max(length - (self.window_size + 1), 0)

        slice_start_position = max(
            (self.window_size + 1) - length, 0
        )

        slice_end_position = slice_start_position + 2 * length - 1 

        if pad_length > 0:
            relative_embeddings = F.pad(
                relative_embeddings, [0, 0, pad_length, pad_length]
            )


        return relative_embeddings[ :, slice_start_position:slice_end_position ]
    

    def forward(self, x, c, attn_mask=None):
        """
        x = query/question      \n
        c = context/relevance   \n

        Self-attention:
        x = c
        The sequence examines itself.
        
        Cross-attention:
        x ≠ c
        One sequence examines another sequence.
        """
        q = self.conv_q(x)
        k = self.conv_k(c)
        v = self.conv_v(c)

        x, self.attn = self.attention(q, k , v, mask=attn_mask)

        x = self.conv_o(x)

        return x
    
    def attention(self, query, key, value, mask=None):

        b, d, t_s = key.size()       #b - batch size, d - channel size,   t_s - context length
        t_t = query.size(2)

        query = query.view(
            b, self.n_heads, self.k_channels, t_t
        ).transpose(2, 3)

        key = key.view(
            b, self.n_heads, self.k_channels, t_s
        ).transpose(2, 3)

        value = value.view(
            b, self.n_heads, self.k_channels, t_s
        ).transpose(2, 3)
        
        scores = torch.matmul(query, key.transpose(-2, -1))  / math.sqrt(self.k_channels)

        if self.window_size is not None:
            key_relative_embeddings = self._get_relative_embeddings(
                self.emb_rel_k, t_s
            )

            relative_logits = torch.matmul(
                query, key_relative_embeddings.transpose(-2, -1)
            )

            relative_logits = self._relative_position_to_absolute_position(relative_logits)

            scores = scores + relative_logits / math.sqrt(self.k_channels)

        if mask is not None:
            scores = scores.masked_fill(mask==0, -1e4)

        p_attn = F.softmax(scores, dim=-1)
        p_attn = self.drop(p_attn)


        output = torch.matmul(p_attn, value)

        if self.window_size is not None:
            relative_weights = self._absolute_position_to_relative_position(
                p_attn
            )

            value_relative_weights = self._get_relative_embeddings(self.emb_rel_v, t_s)

            output = output + torch.matmul(
                relative_weights, value_relative_weights
            )

        output = output.transpose(2, 3).contiguous().view(
            b, d, t_t
        )

        return output, p_attn
    

class FFN(nn.Module):

    def __init__(self,
        in_channels,
        out_channels,
        filter_channels,
        kernel_size,
        p_dropout=0.0,
        activation=None
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.filter_channels = filter_channels
        self.kernel_size = kernel_size
        self.activation = activation

        self.conv_1 = nn.Conv1d(
            in_channels,
            filter_channels,
            kernel_size
        )

        self.conv_2 = nn.Conv1d(
            filter_channels,
            out_channels,
            kernel_size
        )

        self.drop = nn.Dropout(p_dropout)


    def _get_padding(self, x, kernel_size):
        if kernel_size == 1: return x

        pad = (kernel_size - 1)//2

        return F.pad(x, (pad, pad))
    
    def forward(self, x, x_mask):

        x = self._get_padding(x * x_mask, self.kernel_size)

        x = self.conv_1(x)

        if self.activation == "gelu":
            x = x * torch.sigmoid(1.702 * x)

        else: x = F.relu(x)


        x = self.drop(x)

        x = self._get_padding(x * x_mask,  self.kernel_size)
        x = self.conv_2(x)

        return x * x_mask
    

class Encoder(nn.Module):

    def __init__(self,
        hidden_channels,
        filter_channels,
        n_heads,
        n_layers,
        kernel_size=1,
        p_dropout=0.0,
        window_size=10
    ):
        super().__init__()

        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.window_size = window_size

        self.drop = nn.Dropout(p_dropout)

        self.attn_layers = nn.ModuleList()
        self.norm_layers_1 = nn.ModuleList()
        self.ffn_layers = nn.ModuleList()
        self.norm_layers_2 = nn.ModuleList()

        for _ in range(n_layers):
            self.attn_layers.append(
                MultiHeadAttention(
                    hidden_channels,
                    hidden_channels,
                    n_heads,
                    p_dropout=p_dropout,
                    window_size=window_size
                )
            )

            self.norm_layers_1.append(
                LayerNorm(hidden_channels)
            )

            self.ffn_layers.append(
                FFN(
                    hidden_channels,
                    hidden_channels,
                    filter_channels,
                    kernel_size,
                    p_dropout=p_dropout
                )
            )

            self.norm_layers_2.append(
                LayerNorm(hidden_channels)
            )


    def forward(self, x, x_mask):
        
        attn_mask = x_mask.unsqueeze(2) * x_mask.unsqueeze(-1)
        x = x * x_mask

        for i in range(self.n_layers):
            y = self.attn_layers[i](
                x, x, attn_mask
            )

            y = self.drop(y)

            x = self.norm_layers_1[i](x + y)
            y = self.ffn_layers[i](x, x_mask)
            y = self.drop(y)

            x = self.norm_layers_2[i](x + y)
            
        return x * x_mask