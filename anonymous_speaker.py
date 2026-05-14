import torch
import torchaudio
import torchaudio.functional as F
from speechbrain.inference.speaker import EncoderClassifier
from tqdm.auto import tqdm

from src.utils import is_iterable_not_string


def generate_speaker_embedding(input_audio_paths, device):
    if not is_iterable_not_string(input_audio_paths):
        input_audio_paths = [input_audio_paths]

    print('Loading Speaker Model')
    speaker_model = EncoderClassifier.from_hparams(
        source='speechbrain/spkrec-ecapa-voxceleb',
        savedir='pretrained_models/spkrec-ecapa-voxceleb',
        run_opts={'device': device},
    ).to(device)
    speaker_model.eval()
    for param in speaker_model.parameters():
        param.requires_grad = False

    print(f'Extracting embeddings from {len(input_audio_paths)} audios')
    embedding_sum = torch.zeros((1, 192), device=device)
    with torch.no_grad():
        for audio_path in tqdm(input_audio_paths):
            wav, sr = torchaudio.load(audio_path)

            if sr != 16000:
                wav = F.resample(wav, sr, 16000)

            # Extract embedding: [1, 192]
            speaker_emb = speaker_model.encode_batch(wav.to(device)).squeeze(1)
            embedding_sum += speaker_emb

    pseudo_embedding = embedding_sum / len(input_audio_paths)

    return pseudo_embedding


if __name__ == '__main__':
    from glob import glob

    all_voice_paths = sorted(glob('./data/VCTK_preprocessed/*.wav'))[0:75000]

    device = 'cuda:0' if torch.cuda.is_available() else 'xpu' if torch.xpu.is_available() else 'cpu'

    fake_speaker = generate_speaker_embedding(all_voice_paths, device=device)
    torch.save(fake_speaker, 'anonymous_speaker.pt')

    print('Saved to anonymous_speaker_01.pt')
