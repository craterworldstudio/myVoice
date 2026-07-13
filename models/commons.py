import torch

def sequence_mask(lengths, max_length=None):

    if max_length is None:
        max_length = lengths.max()


    x = torch.arange(
        max_length,
        dtype=   lengths.dtype,
        device=  lengths.device
    )

    return x.unsqueeze(0) < lengths.unsqueeze(1)


def fused_add_tanh_sigmoid_multiply(a, b, channels):
    channels_int = channels[0]

    in_act = a + b

    in_tanh = torch.tanh(in_act[:, :channels_int, :])
    in_sigmoid = torch.sigmoid(in_act[:, channels_int:, :])

    return in_tanh * in_sigmoid


def get_padding(kernel_size, dilation=1):
    return int((kernel_size * dilation - dilation) / 2)