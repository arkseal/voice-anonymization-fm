from pathlib import Path

import torch
import torchaudio
from speechbrain.inference.speaker import EncoderClassifier
from transformers import HubertModel, SpeechT5HifiGan

from src.data import VCTKTransform
from src.flow import _generate
from src.model import FlowMatchingUNet


def generate(
    input_audio_path,
    speaker_emb_path,
    model_path,
    device,
    save_path=Path('./results.wav'),
):
    print('Loading Models...')
    hubert_model = HubertModel.from_pretrained('facebook/hubert-base-ls960').to(device)
    hubert_model.eval()
    for param in hubert_model.parameters():
        param.requires_grad = False

    speaker_model = EncoderClassifier.from_hparams(
        source='speechbrain/spkrec-ecapa-voxceleb',
        savedir='pretrained_models/spkrec-ecapa-voxceleb',
        run_opts={'device': device},
    ).to(device)
    speaker_model.eval()
    for param in speaker_model.parameters():
        param.requires_grad = False

    print('Loading Hifi-GAN')
    vocoder = SpeechT5HifiGan.from_pretrained('microsoft/speecht5_hifigan').to(device)

    print('Loading U Net Model')
    preprocessing = VCTKTransform()
    model = FlowMatchingUNet()

    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict['model_state_dict'])

    model.to(device)
    model.eval()
    print('Loaded Model')

    print('Loading Input Audio and Speaker Embedding')
    input_audio, sr = torchaudio.load(input_audio_path)
    input_audio.squeeze_(0)
    speaker_emb = torch.load(speaker_emb_path, map_location=device)

    with torch.no_grad():
        hub_au, aud, _ = preprocessing((input_audio, sr))
        hub_au, aud = hub_au.to(device), aud.to(device)
        hub_au.unsqueeze_(0)
        aud.unsqueeze_(0)

        content_emb = hubert_model(hub_au).last_hidden_state.transpose(1, 2)
        # speaker_emb = speaker_model.encode_batch(aud).squeeze(1)

    print('Generating Sample...')
    waveform, generated_mel = _generate(
        model, content_emb, speaker_emb, vocoder, device, leave_progress=True
    )
    waveform = waveform.cpu()

    torchaudio.save(save_path, waveform, 16000)
    torch.save(generated_mel, save_path.parent / f'{save_path.stem}.pt')


if __name__ == '__main__':
    generate(
        './data/VCTK_preprocessed/77000.wav',
        './anonymous_speaker.pt',
        './checkpoints/model_final.pth',
        'cuda:0',
    )
