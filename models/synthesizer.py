import math, torch
from torch import nn

from models.modules import ResidualCouplingLayer, Flip
from models.hifigan import Generator
from models.attentions import Encoder
from models.commons import sequence_mask


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
    

class TextEncoder(nn.Module):

    def __init__(
        self,
        out_channels,
        hidden_channels,
        filter_channels,
        n_heads,
        n_layers,
        kernel_size,
        p_dropout
    ):
        super().__init__()

        self.out_channels = out_channels
        self.hidden_channels = hidden_channels

        self.emb_phone = nn.Linear(768, hidden_channels)

        self.emb_pitch = nn.Embedding(
            256,
            hidden_channels
        )

        self.encoder = Encoder(
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers,
            kernel_size,
            p_dropout
        )

        self.proj = nn.Conv1d(
            hidden_channels,
            out_channels * 2,
            1
        )

    def forward(self, phone, pitch, lengths):

        #print("phone in encoder:", phone.shape)
        #print("pitch in encoder:", pitch.shape)
                
        
        x = self.emb_phone(phone)
        #print("phone emb:", x.shape)
        x = x + self.emb_pitch(pitch)

        
        #print("pitch emb:", x.shape)


        x = x.transpose(1, 2)
        x_mask = sequence_mask(lengths, x.size(2)).unsqueeze(1).to(dtype=x.dtype)

        x = x * x_mask
        x  = self.encoder(x, x_mask)

        stats = self.proj(x) * x_mask

        m, logs = torch.split(
            stats, [self.out_channels, self.out_channels], dim=1
        )
        #print("TENC M ",m.min(), m.max())
        #print("TENC LOGS ",logs.min(), logs.max())
        #print("TENC M.ISNAN ",torch.isnan(m).any())
        #print("TENC LOGS.ISNAN ",torch.isnan(logs).any())

        return x, m, logs, x_mask

class SynthesizerTrn(nn.Module):

    def __init__(
            self,
            spec_channels,
            segment_size,
            inter_channels,
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers,
            kernel_size,
            p_dropout,
            resblock,
            resblock_kernel_sizes,
            resblock_dilation_sizes,
            upsample_rates,
            upsample_initial_channel,
            upsample_kernel_sizes,
            spk_embed_dim,
            gin_channels,
            sr,
            **kwargs
            ):
        super().__init__()

        self.enc_p = TextEncoder(
            inter_channels, hidden_channels, filter_channels,
            n_heads, n_layers, kernel_size, p_dropout
        )

        self.dec = Generator(
            inter_channels, 
            resblock_kernel_sizes, resblock_dilation_sizes,
            upsample_rates, upsample_initial_channel, upsample_kernel_sizes,
            gin_channels=gin_channels, sample_rate=sr
        )

        self.flow = ResidualCouplingBlock(
            inter_channels, hidden_channels, 
            5, 1, 3,
            gin_channels=gin_channels,
        )

        self.emb_g = nn.Embedding(
            spk_embed_dim,
            gin_channels
        )

        self.segment_size = segment_size
        self.inter_channels = inter_channels
        self.spec_channels = spec_channels
        self.resblock = resblock


    @torch.no_grad()
    def infer(
        self,
        phone,
        phone_lengths,
        pitch,
        pitchf,
        sid,
    ):
        g = self.emb_g(sid).unsqueeze(-1)
        _, m_p, logs_p, x_mask = self.enc_p(
            phone,
            pitch,
            phone_lengths
        )
        
        z_p = m_p + torch.randn_like(m_p) * torch.exp(logs_p)
        z = self.flow(
            z_p, x_mask,
            g=g,
            reverse=True
        )
        o = self.dec(
            z * x_mask,
            pitchf,
            g=g
        )

        print("m_p", m_p.min(), m_p.max())
        print("logs_p", logs_p.min(), logs_p.max())



        print(
            "Z:",
            z.min().item(),
            z.max().item(),
            z.mean().item(),
        )

        print(
            "OUT:",
            o.min().item(),
            o.max().item(),
            o.mean().item(),
        )

        return o
