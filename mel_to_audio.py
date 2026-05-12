import sys
from pathlib import Path

import torch
import torchaudio
from transformers import SpeechT5HifiGan

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage: python ./mel_to_audio.py [mel.pt]')
        exit(1)

    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.xpu.is_available():
        device = 'xpu'
    else:
        device = 'cpu'

    mel_file = Path(sys.argv[1])
    assert mel_file.exists(), f'{mel_file} does not exist'

    print('Loading Hifi-GAN')
    vocoder = SpeechT5HifiGan.from_pretrained('microsoft/speecht5_hifigan').to(device)

    mel = torch.load(mel_file).to(device).transpose(1, 2)
    print(mel.shape)

    with torch.no_grad():
        waveform = vocoder(mel)

    output_waveform = waveform[0].unsqueeze(0).cpu()

    torchaudio.save(mel_file.parent / f'{mel_file.stem}.wav', output_waveform, sample_rate=16000)
