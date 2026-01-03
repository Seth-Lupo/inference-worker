"""Mel spectrogram utilities from Matcha-TTS - inlined."""
import torch
import librosa

_mel_basis_cache = {}
_hann_window_cache = {}


def mel_spectrogram(y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center=False):
    """Compute mel spectrogram from waveform.

    This is a compatibility function that replaces matcha.utils.audio.mel_spectrogram.
    Used for training GAN-based vocoders.
    """
    global _mel_basis_cache, _hann_window_cache

    if isinstance(y, torch.Tensor):
        device = y.device
    else:
        device = torch.device('cpu')
        y = torch.from_numpy(y).float()

    cache_key = f"{n_fft}_{num_mels}_{sampling_rate}_{fmin}_{fmax}_{device}"
    if cache_key not in _mel_basis_cache:
        mel = librosa.filters.mel(sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax)
        _mel_basis_cache[cache_key] = torch.from_numpy(mel).float().to(device)

    window_key = f"{win_size}_{device}"
    if window_key not in _hann_window_cache:
        _hann_window_cache[window_key] = torch.hann_window(win_size).to(device)

    mel_basis = _mel_basis_cache[cache_key]
    hann_window = _hann_window_cache[window_key]

    y = y.unsqueeze(0) if y.dim() == 1 else y
    y = torch.nn.functional.pad(
        y, (int((n_fft - hop_size) / 2), int((n_fft - hop_size) / 2)), mode="reflect"
    )

    spec = torch.stft(
        y,
        n_fft,
        hop_length=hop_size,
        win_length=win_size,
        window=hann_window,
        center=center,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
        return_complex=True,
    )
    spec = torch.abs(spec)

    mel = torch.matmul(mel_basis, spec)
    mel = torch.log(torch.clamp(mel, min=1e-5))

    return mel
