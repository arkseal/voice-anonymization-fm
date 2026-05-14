import torch
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from .VCTK import VCTK, VCTKTransform


def pad_collat_fn(batch):
    hubert_waveforms = [item[0] for item in batch]
    raw_waveforms = [item[1] for item in batch]
    mel_specs = [
        item[2].squeeze(0).transpose(0, 1) if item[2].dim() == 3 else item[2].transpose(0, 1)
        for item in batch
    ]

    padded_hubert = pad_sequence(hubert_waveforms, batch_first=True)
    padded_waveforms = pad_sequence(raw_waveforms, batch_first=True)
    # Padd mels with -11.5129 for silence
    padded_mels = pad_sequence(mel_specs, batch_first=True, padding_value=-11.5129).transpose(1, 2)

    mel_time_len = padded_mels.shape[-1]
    if mel_time_len % 4 != 0:
        pad_amount = 4 - (mel_time_len % 4)
        # Pad last dim by (0 left, pad_amount right) with mel silence value
        padded_mels = F.pad(padded_mels, (0, pad_amount), value=-11.5129)

        # Pad waveform samples
        waveform_pad = pad_amount * 256
        padded_hubert = F.pad(padded_hubert, (0, waveform_pad))
        padded_waveforms = F.pad(padded_waveforms, (0, waveform_pad))

    return padded_hubert, padded_waveforms, padded_mels


def get_train_data(length=None, batch_size=8, num_workers=0):
    transform = VCTKTransform()

    dataset = VCTK(length if length is not None else 1024, transform=transform)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=pad_collat_fn,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
    )

    return dataloader
