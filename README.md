# Voice Anonymization Flow Matching

# TODO
- [ ] Add wandb support for training visualization and logging
- [ ] Add audio support
    - [ ] Find an audio dataset to train on (VCTK or LibriSpeech), convert to PyTorch dataset
    - [ ] Feature extraction for audio (spectrograms -> 80-dimensional log Mel spectrogram)
    - [ ] Feature reconstruction to audio (inverse spectrograms), possibly HiFi-GAN
- [ ] Handle audio in model architecture
    - [ ] Handle varying sequence lengths 
    - [ ] Content Conditioning, use HuBERT or Wav2Vec2 to extract content features and condition the model on them
    - [ ] Speaker Conditioning, use speaker embeddings to condition the model on speaker identity
- [ ] Train model on reconstructing speech
- [ ] Train model on anonymizing speech
- [ ] Add evaluation metrics for anonymization (EER, WER, etc.)

## Getting Started

Clone the repository and set up the python environment (Python 3.14.2 was used).

```bash
git clone https://github.com/arkseal/voice-anonymization-fm.git
cd voice-anonymization-fm
```

Create a virtual environment and install the required packages:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

https://datashare.ed.ac.uk/handle/10283/3443

# THE FOLLOWING NEEDS TO BE UPDATED

## Datasets

This implementation used MINIST dataset for training and evaluation. The code will be modified to support other datasets in the future.

## Training
To train the model, run the following command:

```bash
python main.py --train
```

## Generation
To generate samples from the model, run the following command:

```bash
python main.py --generate
```

## References

- [1] Lipman, Yaron, et al. "Flow Matching for Generative Modeling." [arXiv:2210.02747](https://arxiv.org/abs/2210.02747)
- [2] Ronneberger, Olaf et al. "U-Net: Convolutional Networks for Biomedical Image Segmentation." [arXiv:1505.04597](https://arxiv.org/pdf/1505.04597)
- [3] Janupalli, Pranay. "Understanding Sinusoidal Positional Encoding in Transformers." [Medium](https://medium.com/@pranay.janupalli/understanding-sinusoidal-positional-encoding-in-transformers-26c4c161b7cc)
- [4] [keishihara/flow-matching](https://github.com/keishihara/flow-matching)