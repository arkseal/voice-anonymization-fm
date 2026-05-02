from glob import glob
from pathlib import Path

import torch
import torchaudio
import torchaudio.transforms as T
from tqdm import tqdm


def preprocess_VCTK():
    new_path = Path('./data/VCTK_preprocessed/')
    if not new_path.exists():
        new_path.mkdir()

    audios = sorted(list(glob('./data/VCTK-Corpus-0.92/**/*.flac', recursive=True)))
    resampler = T.Resample(orig_freq=48000, new_freq=16000)

    for i, audio_path in enumerate(tqdm(audios)):
        audio, _ = torchaudio.load(audio_path)
        resampled_audio = resampler(audio)

        torchaudio.save(
            new_path / (str(i).zfill(len(str(len(audios)))) + '.wav'),
            resampled_audio,
            16000,
        )


if __name__ == '__main__':
    preprocess_VCTK()
