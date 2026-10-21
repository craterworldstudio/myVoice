import torch
from torch import nn
from torch.nn import functional as F

from models.commons import get_padding

#FSP -/ HifiGAN format

class ResBlock1(nn.Module):

    def __init__(self,
                channels, 
                kernel_size=3, 
                dilation=(1, 3, 5)
                ):
        
        super().__init__()

        self.convs1 = nn.ModuleList()
        self.convs2 = nn.ModuleList()

        for d in dilation:
            conv1 = nn.Conv1d(
                channels,
                channels,
                kernel_size,
                1,
                dilation=d,
                padding=get_padding(kernel_size, d)
            )

            conv2 = nn.Conv1d(
                channels,
                channels,
                kernel_size,
                dilation=1,
                padding=get_padding(kernel_size, 1)
            )


            self.convs1.append(nn.utils.weight_norm(conv1))
            self.convs2.append(nn.utils.weight_norm(conv2))

    def forward(self, x):

        for conv1, conv2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, 0.1)
            xt = conv1(xt)

            xt = F.leaky_relu(xt, 0.1)
            xt = conv2(xt)

            x = xt + x
        
        return x
    

class SineGenerator(nn.Module):

    def __init__(
        self,
        sample_rate,
        harmonic_num=0,
        sine_amp=0.1,
        noise_std=0.003,
        voiced_threshold=0,
    ):
        super().__init__()

        self.sample_rate = sample_rate
        self.harmonic_num = harmonic_num
        self.sine_amp = sine_amp
        self.noise_std = noise_std
        self.voiced_threshold = voiced_threshold

        self.dim = harmonic_num + 1


    def _f0_to_uv(self, f0):
        return (f0 > self.voiced_threshold).float()


    def forward(self, f0, upp):
        """
        Parameters
        ----------
        f0 : (B,T)
        upp : total upsampling factor (320)
        """

        with torch.no_grad():

            f0 = f0[:, :, None]

            f0_buf = torch.zeros(
                f0.size(0),
                f0.size(1),
                self.dim,
                device=f0.device,
                dtype=f0.dtype,
            )

            # Fundamental
            f0_buf[:, :, 0] = f0[:, :, 0]

            # Harmonics
            for i in range(self.harmonic_num):
                f0_buf[:, :, i + 1] = f0[:, :, 0] * (i + 2)

            # Cycles per frame
            rad_values = (f0_buf / self.sample_rate) % 1.0

            # Random initial phase (except fundamental)
            rand_ini = torch.rand(
                f0_buf.size(0),
                self.dim,
                device=f0.device,
                dtype=f0.dtype,
            )

            rand_ini[:, 0] = 0

            rad_values[:, 0, :] += rand_ini

            cumulative = torch.cumsum(rad_values, dim=1)

            cumulative *= upp

            cumulative = F.interpolate(
                cumulative.transpose(1, 2),
                scale_factor=float(upp),
                mode="linear",
                align_corners=True,
            ).transpose(1, 2)

            rad_values = F.interpolate(
                rad_values.transpose(1, 2),
                scale_factor=float(upp),
                mode="nearest",
            ).transpose(1, 2)

            cumulative %= 1.0

            overflow = (
                cumulative[:, 1:, :] -
                cumulative[:, :-1, :]
            ) < 0

            shift = torch.zeros_like(rad_values)

            shift[:, 1:, :] = overflow * -1.0

            sine = torch.sin(
                torch.cumsum(
                    rad_values + shift,
                    dim=1
                ) * (2 * torch.pi)
            )

            sine *= self.sine_amp

            uv = self._f0_to_uv(f0)

            uv = F.interpolate(
                uv.transpose(1, 2),
                scale_factor=float(upp),
                mode="nearest",
            ).transpose(1, 2)

            noise_amp = (
                uv * self.noise_std
                + (1.0 - uv) * self.sine_amp / 3.0
            )

            noise = noise_amp * torch.randn_like(sine)

            sine = sine * uv + noise

        return sine, uv, noise


class SourceModuleHnNSF(nn.Module):

    def __init__(self, 
                sample_rate,
                harmonic_num=0,
                sine_amp=0.1,
                noise_std=0.003,
                voiced_threshold=0
                ):
        super().__init__()

        self.sine_generator = SineGenerator(
            sample_rate,
            harmonic_num,
            sine_amp,
            noise_std,
            voiced_threshold
        )

        self.l_linear = nn.Linear(harmonic_num + 1, 1)

        self.l_tanh = nn.Tanh()


    def forward(self, f0, upp):

        sine, uv, noise = self.sine_generator( f0, upp )

        source = self.l_linear(sine)
        source = self.l_tanh(source)

        return source, noise, uv

class Generator(nn.Module):
    def __init__(self,
                initial_channel,
                resblock_kernel_sizes,
                resblock_dilation_sizes,
                upsample_rates,
                upsample_initial_channel,
                upsample_kernel_sizes,
                gin_channels=0,
                sample_rate=32000
            ):
        
        super().__init__()

        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)

        self.m_source = SourceModuleHnNSF(
            sample_rate=sample_rate,
            harmonic_num=0
            )

        self.conv_pre = nn.Conv1d(
            initial_channel,
            upsample_initial_channel,
            7, 1, padding=3
            )
        
        self.ups = nn.ModuleList()
        self.noise_convs = nn.ModuleList()
        self.resblocks = nn.ModuleList()

        self.upp = 1

        for u in upsample_rates:
            self.upp *= u

        """
        initial_channel         = 192
        upsample_initial_channel = 512

        upsample_rates          = [10, 8, 2, 2]
        upsample_kernel_sizes   = [20, 16, 4, 4]

        resblock_kernel_sizes   = [3, 7, 11]
        resblock_dilation_sizes = [(1,3,5), (1,3,5), (1,3,5)]

        gin_channels            = 256
        sample_rate             = 32000
        """

        for i, (u, k) in enumerate( zip(upsample_rates, upsample_kernel_sizes)):
            in_channels = upsample_initial_channel // (2** i)
            out_channels = upsample_initial_channel // (2 ** (i+1))

            up = nn.ConvTranspose1d(
                in_channels, out_channels,
                k, u, padding = (k - u) // 2
            )

            self.ups.append(nn.utils.weight_norm(up))

        for i in range(self.num_upsamples):
                
            out_channels = upsample_initial_channel // (2 ** (i + 1))

            if i + 1 < self.num_upsamples:
                stride = 1

                for u in upsample_rates[i + 1:]:
                    stride *= u

                kernel_size = stride * 2

                noise_conv = nn.Conv1d(
                    1,
                    out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=stride // 2
                )

            else:
                noise_conv = nn.Conv1d(
                    1,
                    out_channels,
                    kernel_size=1
                )

            self.noise_convs.append(noise_conv)

        for i in range(self.num_upsamples):

            channels = upsample_initial_channel // (2 ** (i + 1))

            for kernel_size, dilation in zip(
                resblock_kernel_sizes,
                resblock_dilation_sizes
            ):
                self.resblocks.append(
                    ResBlock1(
                        channels,
                        kernel_size=kernel_size,
                        dilation=dilation
                    )
                )

        self.conv_post = nn.Conv1d(
            upsample_initial_channel // (2 ** self.num_upsamples),
            1,
            7,
            1,
            padding=3,
            bias=False
        )

        if gin_channels != 0:
            self.cond = nn.Conv1d(
                gin_channels,
                upsample_initial_channel,
                1
            )


    def forward(self, x, f0, g=None):
        


        """f0 = F.interpolate(
            f0.unsqueeze(1),
            scale_factor=self.upp, # changed self.upp = 320
            mode="linear",
            align_corners=False
        ).transpose(1, 2) """

        har_source, _, _ = self.m_source(f0, self.upp)
        har_source = har_source.transpose(1, 2)

        x = self.conv_pre(x)

        if g is not None and hasattr(self, "cond"):

            x += self.cond(g)

        for i in range(self.num_upsamples):

            x = F.leaky_relu(x, 0.1)
            x = self.ups[i](x)

            source = self.noise_convs[i](har_source)

            x += source

            xs = None

            for j in range(self.num_kernels):
                index = i * self.num_kernels + j
                res = self.resblocks[index](x)

                if xs is None:
                    xs = res
                else:
                    xs = xs+res

            x = xs / self.num_kernels

        x = F.leaky_relu(x, 0.1)
        x=self.conv_post(x)
        x=torch.tanh(x)

        return x



