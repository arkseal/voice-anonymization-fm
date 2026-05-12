import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm, trange

# SpeechT5 Log-Mel Spectrogram
MEL_MEAN = -5.5
MEL_STD = 2.0


def compute_loss(model, x1, content_emb, speaker_emb, device='cpu'):
    b, c, t_frame = x1.shape

    x1_norm = (x1 - MEL_MEAN) / MEL_STD

    x0 = torch.randn_like(x1_norm).to(device)

    t = torch.rand(b).to(device)
    t_exp = t[..., None, None]  # or t.view(b, 1, 1)

    # xt = (1 - t) x0 + t x1
    xt = (1 - t_exp) * x0 + t_exp * x1_norm
    ut = x1_norm - x0  # actual

    vt = model(xt, t, content_emb, speaker_emb)  # pred

    loss = F.mse_loss(vt, ut)
    return loss


@torch.no_grad()
def sample_ode(
    model,
    shape,
    content_emb,
    speaker_emb,
    device='cpu',
    steps=20,
    leave_progress=False,
    store_all=False,
):
    x = torch.randn(shape).to(device)
    dt = 1 / steps

    if store_all:
        all_mels = []

    for i in trange(steps, leave=leave_progress):
        t_val = i / steps
        t = torch.full([shape[0]], t_val, device=device)

        vt = model(x, t, content_emb, speaker_emb)
        x += vt * dt

        if store_all:
            all_mels.append(x.cpu().clone().detach())

    if store_all:
        a = torch.stack(all_mels)
        return a
    return x


@torch.no_grad()
def _generate(
    model,
    content_emb,
    speaker_emb,
    vocoder,
    device,
    steps=20,
    leave_progress=False,
    store_all=False,
):
    assert not store_all, 'TODO: need to implement store_all if needed'

    batch_size = content_emb.shape[0]

    # Scale HuBERT frames (1 every 320 samples) to Mel frames (1 every 256 samples)
    time_frames = int(content_emb.shape[-1] * (320 / 256))

    # Make divisible by 4 for U-Net downsampling
    if time_frames % 4 != 0:
        time_frames += 4 - (time_frames % 4)

    shape = (batch_size, 80, time_frames)

    generated_mels_norm = sample_ode(
        model,
        shape,
        content_emb,
        speaker_emb,
        device=device,
        steps=steps,
        leave_progress=leave_progress,
        store_all=store_all,
    )

    generated_mels = (generated_mels_norm * MEL_STD) + MEL_MEAN

    if vocoder is not None:
        hifigan_inputs = generated_mels.transpose(1, 2)
        waveform = vocoder(hifigan_inputs)

        return waveform, generated_mels

    return None, generated_mels
