from pathlib import Path

import torch
import torchaudio
import torchaudio.transforms as T
from transformers import Wav2Vec2FeatureExtractor


class VCTK(torch.utils.data.Dataset):
    def __init__(
        self,
        length=88328,
        start=0,
        data_src='./data/VCTK_preprocessed/',
        transform=None,
    ):
        super().__init__()
        assert start >= 0 and start < 88328, f'Start {start} is not in between 0 and 88328'
        assert length <= 88328 - start, f'Length {length} is greater than {88328 - start}'
        self.start = start
        self.length = length
        self.src = Path(data_src)
        self.transform = transform

    def __getitem__(self, index):
        a = torchaudio.load(self.src / (str(index + self.start).zfill(5) + '.wav'))
        if self.transform is not None:
            a = self.transform(a)
        return a

    def __len__(self):
        return self.length


class VCTKTransform:
    def __init__(self, target_sr=16000):
        self.target_sr = target_sr
        self.mel_transform = T.MelSpectrogram(
            sample_rate=target_sr,
            n_fft=1024,
            win_length=1024,
            hop_length=256,
            f_min=0,
            f_max=8000,
            n_mels=80,
            normalized=True,
        )
        self.hubert_processor = Wav2Vec2FeatureExtractor.from_pretrained(
            'facebook/hubert-base-ls960'
        )

    def __call__(self, audio_tuple):
        waveform, sr = audio_tuple

        # Resample if needed
        if sr != self.target_sr:
            resampler = T.Resample(orig_freq=sr, new_freq=self.target_sr)
            waveform = resampler(waveform)

        mel_spec = self.mel_transform(waveform).squeeze(0)

        wav_np = waveform.squeeze(0).numpy()
        hubert_wav = self.hubert_processor(
            wav_np, sampling_rate=self.target_sr, return_tensors='pt'
        ).input_values.squeeze(0)

        raw_wave = waveform.squeeze(0)

        return hubert_wav, raw_wave, mel_spec
