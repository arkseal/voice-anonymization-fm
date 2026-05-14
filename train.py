from pathlib import Path

import flair
import torch
import torch.optim as optim
import torchaudio
import torchcodec
import transformers
from speechbrain.inference.speaker import EncoderClassifier
from torch.amp import GradScaler, autocast
from tqdm.auto import tqdm, trange
from transformers import HubertModel

from src.data import get_train_data
from src.flow import _generate, compute_loss
from src.model import FlowMatchingUNet
from src.utils import get_precision_dtype


def train(
    length=None,
    batch_size=8,
    num_workers=2,
    lr=1e-4,
    epochs=25,
    device='cpu',
    checkpoint_path=Path('./checkpoints'),
    save_path=Path('./results'),
    precision='full',
    resume_path=None,
):
    dataloader = get_train_data(length=length, batch_size=batch_size, num_workers=num_workers)

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

    model = FlowMatchingUNet().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    scaler = GradScaler(device) if ('amp') in precision else None
    autocast_dtype = get_precision_dtype(precision)

    start_epoch = 1
    if resume_path:
        checkpoint = torch.load(resume_path)

        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if scaler and 'scaler_state_dict' in checkpoint:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])

        start_epoch = checkpoint['epoch'] + 1
        epochs = epochs + start_epoch - 1
        loss = checkpoint['loss']

    for epoch in trange(start_epoch, epochs + 1):
        model.train()
        total_loss = 0

        for hub_au, aud, mel in tqdm(dataloader, desc=f'Epoch {epoch}', leave=False):
            hub_au, mel = hub_au.to(device, non_blocking=True), mel.to(device, non_blocking=True)
            optimizer.zero_grad()

            with torch.no_grad():
                content_emb = hubert_model(hub_au).last_hidden_state.transpose(1, 2)
                speaker_emb = speaker_model.encode_batch(aud).squeeze(1)

            with autocast(device, autocast_dtype):
                loss = compute_loss(model, mel, content_emb, speaker_emb, device=device)

            if scaler:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)

        if epoch % 1 == 0:
            tqdm.write(f'Epoch: {epoch} Loss: {avg_loss:.4f}')
            model.eval()
            tqdm.write(f'Generating Samples for epoch {epoch}:')

            if scaler:
                with autocast(device):
                    _, generated_mel = _generate(
                        model, content_emb[0:1], speaker_emb[0:1], None, device
                    )
            else:
                _, generated_mel = _generate(
                    model, content_emb[0:1], speaker_emb[0:1], None, device
                )

            mel_save_path = save_path / f'epoch_{epoch:03d}.pt'
            torch.save(generated_mel, mel_save_path)

            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
            }
            if scaler:
                checkpoint['scaler_state_dict'] = scaler.state_dict()
            torch.save(checkpoint, checkpoint_path / f'model_epoch_{epoch}.pth')

    print('Training complete')
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    if scaler:
        checkpoint['scaler_state_dict'] = scaler.state_dict()
    torch.save(checkpoint, checkpoint_path / f'model_epoch_{epoch}.pth')

    final_model = {'model_state_dict': model.state_dict()}
    torch.save(final_model, checkpoint_path / 'model_final.pth')


if __name__ == '__main__':
    train(75000, batch_size=16, num_workers=16, epochs=10, device='cuda:0', precision='amp_bf16')
