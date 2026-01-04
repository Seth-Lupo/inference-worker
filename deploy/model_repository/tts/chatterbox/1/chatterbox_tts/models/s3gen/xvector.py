#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
# Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights Reserved.
#  MIT License  (https://opensource.org/licenses/MIT)
# Modified from 3D-Speaker (https://github.com/alibaba-damo-academy/3D-Speaker)


from collections import OrderedDict
import math
import torch
import torch.nn.functional as F
import torch.utils.checkpoint as cp


def _hz_to_mel(freq: float) -> float:
    """Convert Hz to Mel scale."""
    return 2595.0 * math.log10(1.0 + freq / 700.0)


def _mel_to_hz(mel: float) -> float:
    """Convert Mel scale to Hz."""
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def _create_mel_filterbank(
    num_mel_bins: int,
    num_fft_bins: int,
    sample_rate: int,
    low_freq: float = 20.0,
    high_freq: float = None,
) -> torch.Tensor:
    """Create mel filterbank matrix."""
    if high_freq is None:
        high_freq = sample_rate / 2.0

    # Mel points
    low_mel = _hz_to_mel(low_freq)
    high_mel = _hz_to_mel(high_freq)
    mel_points = torch.linspace(low_mel, high_mel, num_mel_bins + 2)
    hz_points = torch.tensor([_mel_to_hz(m.item()) for m in mel_points])

    # FFT bin frequencies
    fft_bin_width = sample_rate / (2.0 * (num_fft_bins - 1))
    bin_points = (hz_points / fft_bin_width).long()

    # Create filterbank
    filterbank = torch.zeros(num_mel_bins, num_fft_bins)
    for i in range(num_mel_bins):
        left = bin_points[i].item()
        center = bin_points[i + 1].item()
        right = bin_points[i + 2].item()

        for j in range(left, center):
            if center != left:
                filterbank[i, j] = (j - left) / (center - left)
        for j in range(center, right):
            if right != center:
                filterbank[i, j] = (right - j) / (right - center)

    return filterbank


def fbank_pure_torch(
    waveform: torch.Tensor,
    num_mel_bins: int = 80,
    frame_length: float = 25.0,
    frame_shift: float = 10.0,
    sample_frequency: float = 16000.0,
    dither: float = 0.0,
    energy_floor: float = 1.0,
    preemphasis_coefficient: float = 0.97,
) -> torch.Tensor:
    """
    Compute filterbank features (Kaldi-compatible).

    Args:
        waveform: Audio tensor of shape (1, num_samples) or (num_samples,)
        num_mel_bins: Number of mel bins
        frame_length: Frame length in ms
        frame_shift: Frame shift in ms
        sample_frequency: Sample rate
        dither: Dithering coefficient
        energy_floor: Floor on energy
        preemphasis_coefficient: Preemphasis coefficient

    Returns:
        Filterbank features of shape (num_frames, num_mel_bins)
    """
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)

    device = waveform.device
    dtype = waveform.dtype

    # Convert ms to samples
    frame_length_samples = int(frame_length * sample_frequency / 1000.0)
    frame_shift_samples = int(frame_shift * sample_frequency / 1000.0)

    # Round to power of 2 for FFT
    n_fft = 1
    while n_fft < frame_length_samples:
        n_fft *= 2

    # Preemphasis
    if preemphasis_coefficient > 0:
        waveform = torch.cat([
            waveform[:, :1],
            waveform[:, 1:] - preemphasis_coefficient * waveform[:, :-1]
        ], dim=1)

    # Dithering
    if dither > 0:
        waveform = waveform + dither * torch.randn_like(waveform)

    # Frame the signal
    waveform = waveform.squeeze(0)
    num_samples = waveform.shape[0]
    num_frames = max(1, 1 + (num_samples - frame_length_samples) // frame_shift_samples)

    # Pad if necessary
    pad_length = (num_frames - 1) * frame_shift_samples + frame_length_samples - num_samples
    if pad_length > 0:
        waveform = F.pad(waveform, (0, pad_length))

    # Create frames
    frames = waveform.unfold(0, frame_length_samples, frame_shift_samples)

    # Apply window
    window = torch.hann_window(frame_length_samples, device=device, dtype=dtype)
    frames = frames * window

    # Pad to n_fft
    if frame_length_samples < n_fft:
        frames = F.pad(frames, (0, n_fft - frame_length_samples))

    # FFT
    spectrum = torch.fft.rfft(frames)
    power_spectrum = (spectrum.real ** 2 + spectrum.imag ** 2)

    # Apply energy floor
    power_spectrum = torch.clamp(power_spectrum, min=energy_floor)

    # Create mel filterbank
    num_fft_bins = n_fft // 2 + 1
    mel_filterbank = _create_mel_filterbank(
        num_mel_bins, num_fft_bins, int(sample_frequency)
    ).to(device=device, dtype=dtype)

    # Apply mel filterbank
    mel_energies = torch.matmul(power_spectrum, mel_filterbank.T)

    # Log compression
    mel_energies = torch.log(torch.clamp(mel_energies, min=energy_floor))

    return mel_energies


def pad_list(xs, pad_value):
    """Perform padding for the list of tensors.

    Args:
        xs (List): List of Tensors [(T_1, `*`), (T_2, `*`), ..., (T_B, `*`)].
        pad_value (float): Value for padding.

    Returns:
        Tensor: Padded tensor (B, Tmax, `*`).

    Examples:
        >>> x = [torch.ones(4), torch.ones(2), torch.ones(1)]
        >>> x
        [tensor([1., 1., 1., 1.]), tensor([1., 1.]), tensor([1.])]
        >>> pad_list(x, 0)
        tensor([[1., 1., 1., 1.],
                [1., 1., 0., 0.],
                [1., 0., 0., 0.]])

    """
    n_batch = len(xs)
    max_len = max(x.size(0) for x in xs)
    pad = xs[0].new(n_batch, max_len, *xs[0].size()[1:]).fill_(pad_value)

    for i in range(n_batch):
        pad[i, : xs[i].size(0)] = xs[i]

    return pad


def extract_feature(audio):
    features = []
    feature_times = []
    feature_lengths = []
    for au in audio:
        feature = fbank_pure_torch(au.unsqueeze(0), num_mel_bins=80)
        feature = feature - feature.mean(dim=0, keepdim=True)
        features.append(feature)
        feature_times.append(au.shape[0])
        feature_lengths.append(feature.shape[0])
    # padding for batch inference
    features_padded = pad_list(features, pad_value=0)
    # features = torch.cat(features)
    return features_padded, feature_lengths, feature_times


class BasicResBlock(torch.nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicResBlock, self).__init__()
        self.conv1 = torch.nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=(stride, 1), padding=1, bias=False
        )
        self.bn1 = torch.nn.BatchNorm2d(planes)
        self.conv2 = torch.nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = torch.nn.BatchNorm2d(planes)

        self.shortcut = torch.nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = torch.nn.Sequential(
                torch.nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=(stride, 1),
                    bias=False,
                ),
                torch.nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class FCM(torch.nn.Module):
    def __init__(self, block=BasicResBlock, num_blocks=[2, 2], m_channels=32, feat_dim=80):
        super(FCM, self).__init__()
        self.in_planes = m_channels
        self.conv1 = torch.nn.Conv2d(1, m_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(m_channels)

        self.layer1 = self._make_layer(block, m_channels, num_blocks[0], stride=2)
        self.layer2 = self._make_layer(block, m_channels, num_blocks[0], stride=2)

        self.conv2 = torch.nn.Conv2d(
            m_channels, m_channels, kernel_size=3, stride=(2, 1), padding=1, bias=False
        )
        self.bn2 = torch.nn.BatchNorm2d(m_channels)
        self.out_channels = m_channels * (feat_dim // 8)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return torch.nn.Sequential(*layers)

    def forward(self, x):
        x = x.unsqueeze(1)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = F.relu(self.bn2(self.conv2(out)))

        shape = out.shape
        out = out.reshape(shape[0], shape[1] * shape[2], shape[3])
        return out


def get_nonlinear(config_str, channels):
    nonlinear = torch.nn.Sequential()
    for name in config_str.split("-"):
        if name == "relu":
            nonlinear.add_module("relu", torch.nn.ReLU(inplace=True))
        elif name == "prelu":
            nonlinear.add_module("prelu", torch.nn.PReLU(channels))
        elif name == "batchnorm":
            nonlinear.add_module("batchnorm", torch.nn.BatchNorm1d(channels))
        elif name == "batchnorm_":
            nonlinear.add_module("batchnorm", torch.nn.BatchNorm1d(channels, affine=False))
        else:
            raise ValueError("Unexpected module ({}).".format(name))
    return nonlinear


def statistics_pooling(x, dim=-1, keepdim=False, unbiased=True, eps=1e-2):
    mean = x.mean(dim=dim)
    std = x.std(dim=dim, unbiased=unbiased)
    stats = torch.cat([mean, std], dim=-1)
    if keepdim:
        stats = stats.unsqueeze(dim=dim)
    return stats


class StatsPool(torch.nn.Module):
    def forward(self, x):
        return statistics_pooling(x)


class TDNNLayer(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        bias=False,
        config_str="batchnorm-relu",
    ):
        super(TDNNLayer, self).__init__()
        if padding < 0:
            assert (
                kernel_size % 2 == 1
            ), "Expect equal paddings, but got even kernel size ({})".format(kernel_size)
            padding = (kernel_size - 1) // 2 * dilation
        self.linear = torch.nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )
        self.nonlinear = get_nonlinear(config_str, out_channels)

    def forward(self, x):
        x = self.linear(x)
        x = self.nonlinear(x)
        return x


class CAMLayer(torch.nn.Module):
    def __init__(
        self, bn_channels, out_channels, kernel_size, stride, padding, dilation, bias, reduction=2
    ):
        super(CAMLayer, self).__init__()
        self.linear_local = torch.nn.Conv1d(
            bn_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )
        self.linear1 = torch.nn.Conv1d(bn_channels, bn_channels // reduction, 1)
        self.relu = torch.nn.ReLU(inplace=True)
        self.linear2 = torch.nn.Conv1d(bn_channels // reduction, out_channels, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        y = self.linear_local(x)
        context = x.mean(-1, keepdim=True) + self.seg_pooling(x)
        context = self.relu(self.linear1(context))
        m = self.sigmoid(self.linear2(context))
        return y * m

    def seg_pooling(self, x, seg_len=100, stype="avg"):
        if stype == "avg":
            seg = F.avg_pool1d(x, kernel_size=seg_len, stride=seg_len, ceil_mode=True)
        elif stype == "max":
            seg = F.max_pool1d(x, kernel_size=seg_len, stride=seg_len, ceil_mode=True)
        else:
            raise ValueError("Wrong segment pooling type.")
        shape = seg.shape
        seg = seg.unsqueeze(-1).expand(*shape, seg_len).reshape(*shape[:-1], -1)
        seg = seg[..., : x.shape[-1]]
        return seg


class CAMDenseTDNNLayer(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        bn_channels,
        kernel_size,
        stride=1,
        dilation=1,
        bias=False,
        config_str="batchnorm-relu",
        memory_efficient=False,
    ):
        super(CAMDenseTDNNLayer, self).__init__()
        assert kernel_size % 2 == 1, "Expect equal paddings, but got even kernel size ({})".format(
            kernel_size
        )
        padding = (kernel_size - 1) // 2 * dilation
        self.memory_efficient = memory_efficient
        self.nonlinear1 = get_nonlinear(config_str, in_channels)
        self.linear1 = torch.nn.Conv1d(in_channels, bn_channels, 1, bias=False)
        self.nonlinear2 = get_nonlinear(config_str, bn_channels)
        self.cam_layer = CAMLayer(
            bn_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )

    def bn_function(self, x):
        return self.linear1(self.nonlinear1(x))

    def forward(self, x):
        if self.training and self.memory_efficient:
            x = cp.checkpoint(self.bn_function, x)
        else:
            x = self.bn_function(x)
        x = self.cam_layer(self.nonlinear2(x))
        return x


class CAMDenseTDNNBlock(torch.nn.ModuleList):
    def __init__(
        self,
        num_layers,
        in_channels,
        out_channels,
        bn_channels,
        kernel_size,
        stride=1,
        dilation=1,
        bias=False,
        config_str="batchnorm-relu",
        memory_efficient=False,
    ):
        super(CAMDenseTDNNBlock, self).__init__()
        for i in range(num_layers):
            layer = CAMDenseTDNNLayer(
                in_channels=in_channels + i * out_channels,
                out_channels=out_channels,
                bn_channels=bn_channels,
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                bias=bias,
                config_str=config_str,
                memory_efficient=memory_efficient,
            )
            self.add_module("tdnnd%d" % (i + 1), layer)

    def forward(self, x):
        for layer in self:
            x = torch.cat([x, layer(x)], dim=1)
        return x


class TransitLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, bias=True, config_str="batchnorm-relu"):
        super(TransitLayer, self).__init__()
        self.nonlinear = get_nonlinear(config_str, in_channels)
        self.linear = torch.nn.Conv1d(in_channels, out_channels, 1, bias=bias)

    def forward(self, x):
        x = self.nonlinear(x)
        x = self.linear(x)
        return x


class DenseLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, bias=False, config_str="batchnorm-relu"):
        super(DenseLayer, self).__init__()
        self.linear = torch.nn.Conv1d(in_channels, out_channels, 1, bias=bias)
        self.nonlinear = get_nonlinear(config_str, out_channels)

    def forward(self, x):
        if len(x.shape) == 2:
            x = self.linear(x.unsqueeze(dim=-1)).squeeze(dim=-1)
        else:
            x = self.linear(x)
        x = self.nonlinear(x)
        return x

# @tables.register("model_classes", "CAMPPlus")
class CAMPPlus(torch.nn.Module):
    def __init__(
        self,
        feat_dim=80,
        embedding_size=192,
        growth_rate=32,
        bn_size=4,
        init_channels=128,
        config_str="batchnorm-relu",
        memory_efficient=True,
        output_level="segment",
        **kwargs,
    ):
        super().__init__()

        self.head = FCM(feat_dim=feat_dim)
        channels = self.head.out_channels
        self.output_level = output_level

        self.xvector = torch.nn.Sequential(
            OrderedDict(
                [
                    (
                        "tdnn",
                        TDNNLayer(
                            channels,
                            init_channels,
                            5,
                            stride=2,
                            dilation=1,
                            padding=-1,
                            config_str=config_str,
                        ),
                    ),
                ]
            )
        )
        channels = init_channels
        for i, (num_layers, kernel_size, dilation) in enumerate(
            zip((12, 24, 16), (3, 3, 3), (1, 2, 2))
        ):
            block = CAMDenseTDNNBlock(
                num_layers=num_layers,
                in_channels=channels,
                out_channels=growth_rate,
                bn_channels=bn_size * growth_rate,
                kernel_size=kernel_size,
                dilation=dilation,
                config_str=config_str,
                memory_efficient=memory_efficient,
            )
            self.xvector.add_module("block%d" % (i + 1), block)
            channels = channels + num_layers * growth_rate
            self.xvector.add_module(
                "transit%d" % (i + 1),
                TransitLayer(channels, channels // 2, bias=False, config_str=config_str),
            )
            channels //= 2

        self.xvector.add_module("out_nonlinear", get_nonlinear(config_str, channels))

        if self.output_level == "segment":
            self.xvector.add_module("stats", StatsPool())
            self.xvector.add_module(
                "dense", DenseLayer(channels * 2, embedding_size, config_str="batchnorm_")
            )
        else:
            assert (
                self.output_level == "frame"
            ), "`output_level` should be set to 'segment' or 'frame'. "

        for m in self.modules():
            if isinstance(m, (torch.nn.Conv1d, torch.nn.Linear)):
                torch.nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # (B,T,F) => (B,F,T)
        x = self.head(x)
        x = self.xvector(x)
        if self.output_level == "frame":
            x = x.transpose(1, 2)
        return x

    def inference(self, audio_list):
        speech, speech_lengths, speech_times = extract_feature(audio_list)
        results = self.forward(speech.to(torch.float32))
        return results
