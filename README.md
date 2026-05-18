# Voice Anonymization Flow Matching

This repository implements a flow-matching based speech anonymization model for speaker-conditioned audio generation.

## Status

- [ ] Add wandb support for training visualization and logging
- [x] Add audio support
  - [x] Use VCTK audio dataset and PyTorch data pipeline
  - [x] Extract 80-dimensional log Mel spectrogram features
  - [x] Reconstruct audio from spectrograms using HiFi-GAN
- [x] Handle audio in model architecture
  - [x] Support varying sequence lengths
  - [x] Condition on content features from HuBERT
  - [x] Condition on speaker embeddings from ECAPA-TDNN
- [x] Train model on reconstructing speech
- [x] Create generation path for anonymized speech
- [ ] Add evaluation metrics for anonymization (EER, WER, etc.)

## Requirements

- Python 3.14+
- PyTorch + torchaudio compatible with your system
- `datasets`, `flair`, `transformers`, `speechbrain`, `torchcodec`, `tqdm`

The project includes both `pyproject.toml` for dependency management.

## Setup

```bash
git clone https://github.com/arkseal/voice-anonymization-fm.git
cd voice-anonymization-fm
uv sync --no-dev
```

## Data

This repository expects a preprocessed VCTK dataset in `./data/VCTK_preprocessed/`.
The repository does not include the raw dataset itself.

Use `preprocess_data.py` to convert raw VCTK `.flac` files into 16 kHz `.wav` files in `./data/VCTK_preprocessed/`.

Example:

```bash
python preprocess_data.py
```

The training pipeline currently uses VCTK-style speech data and speaker-conditioning from `speechbrain/spkrec-ecapa-voxceleb`.

## Training

Use `main.py` with `--train` to run model training.

Example:

```bash
python main.py --train --device cuda:0 --batch-size 1 --epoch 10 --lr 1e-4 --checkpoint-path ./checkpoints --save-path ./results
```

Optional training flags:

- `--dataset-length`: number of dataset samples to use (default: 75000)
- `--batch-size`: training batch size (default: 1)
- `--num-workers`: number of data loader workers (default: 0)
- `--lr`: learning rate (default: 1e-4)
- `--epoch`: number of training epochs (default: 1)
- `--precision`: one of `full`, `half`, `amp`, `amp_bf16` (default: `full`)
- `--resume-checkpoint`: checkpoint file to resume training from

## Speaker Embeddings

Generate or load a speaker embedding before running generation.

Example:

```bash
python anonymous_speaker.py
```

This script creates an averaged speaker embedding from VCTK audio and saves it as `anonymous_speaker.pt`.

## Generation

Use `main.py` with `--generate` to anonymize audio.

Example:

```bash
python main.py --generate --input-audio-path ./data/VCTK_preprocessed/77000.wav --speaker-emb-path ./anonymous_speaker.pt --model-path ./checkpoints/model_final.pth --save-path ./results/output.wav --device cuda:0
```

Generation flags:

- `--input-audio-path`: path to the input waveform
- `--speaker-emb-path`: speaker embedding file
- `--model-path`: trained model checkpoint
- `--save-path`: output waveform path
- `--overwrite`: allow overwriting an existing output file

## Convert Generated Mel to Audio

If you want to convert a saved mel spectrogram back into waveform audio, use `mel_to_audio.py`.

Example:

```bash
python mel_to_audio.py ./results/output.pt
```

This script loads `SpeechT5HifiGan`, converts the mel spectrogram to waveform, and saves a `.wav` file alongside the original `.pt` file.

## Notes

- The code currently relies on VCTK preprocessing and a pre-trained HuBERT model for content features.
- Generation uses `SpeechT5HifiGan` as the vocoder to convert generated mel spectrograms back to audio.
- Evaluation metrics such as EER/WER are not yet implemented.

## References

- [1] Lipman, Yaron, et al. "Flow Matching for Generative Modeling." [arXiv:2210.02747](https://arxiv.org/abs/2210.02747)
- [2] Ronneberger, Olaf et al. "U-Net: Convolutional Networks for Biomedical Image Segmentation." [arXiv:1505.04597](https://arxiv.org/pdf/1505.04597)
- [3] Janupalli, Pranay. "Understanding Sinusoidal Positional Encoding in Transformers." [Medium](https://medium.com/@pranay.janupalli/understanding-sinusoidal-positional-encoding-in-transformers-26c4c161b7cc)
- [4] [keishihara/flow-matching](https://github.com/keishihara/flow-matching)
