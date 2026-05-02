import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, emb_dim, denom=10000):
        super(SinusoidalPositionEmbeddings, self).__init__()
        self.emb_dim = emb_dim
        self.denom = denom

    def forward(self, time):
        if time.dim() == 1:
            time = time.unsqueeze(1)

        half_dim = self.emb_dim // 2
        i = torch.arange(half_dim, dtype=torch.float32, device=time.device)
        denominator = self.denom ** (2 * i / self.emb_dim)
        args = time / denominator

        sin_emb = args.sin()
        cos_emb = args.cos()

        # Using stack + flatten to alternate sin and cos to adhere to the paper
        embeddings = torch.stack((sin_emb, cos_emb), dim=-1)
        embeddings = embeddings.flatten(start_dim=-2)

        return embeddings


class Block(nn.Module):
    def __init__(self, in_ch, out_ch, cond_emb_dim):
        super(Block, self).__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv1d(out_ch, out_ch, 3, padding=1)

        self.norm1 = nn.GroupNorm(4, out_ch)
        self.norm2 = nn.GroupNorm(4, out_ch)

        self.activation = nn.GELU()

        self.cond_mlp = nn.Sequential(nn.GELU(), nn.Linear(cond_emb_dim, 2 * out_ch))

        if in_ch != out_ch:
            self.shortcut = nn.Conv1d(in_ch, out_ch, 1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x, time_emb):
        # Could flip to norm -> activate -> conv
        x1 = self.conv1(x)
        x1 = self.norm1(x1)

        cond_emb = self.cond_mlp(time_emb)
        cond_emb = cond_emb[..., None]
        gamma, beta = cond_emb.chunk(2, dim=1)
        x1 = x1 * (1 + gamma) + beta

        x1 = self.activation(x1)

        x1 = self.conv2(x1)
        x1 = self.norm2(x1)
        x1 = self.activation(x1)

        x = self.shortcut(x)

        return x1 + x


class FlowMatchingUNet(nn.Module):
    def __init__(
        self,
        mel_bins=80,
        content_dim=768,
        speaker_dim=192,
        down_channels=[128, 256, 512],
        time_emb_dim=256,
    ):
        super(FlowMatchingUNet, self).__init__()
        # SHAPE: (batch, mel_bins=80, time)
        self.mel_bins = mel_bins
        self.content_dim = content_dim
        self.speaker_dim = speaker_dim

        self.down_channels = down_channels  # Increasing depth
        self.time_emb_dim = time_emb_dim
        self.cond_emb_dim = time_emb_dim + speaker_dim

        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.GELU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        # initial projection
        self.conv0 = nn.Conv1d(mel_bins + content_dim, down_channels[0], 3, padding=1)

        # downsample path encoder
        self.downsample = nn.ModuleList(
            [
                nn.Conv1d(down_channels[0], down_channels[1], 4, 2, 1),
                nn.Conv1d(down_channels[1], down_channels[2], 4, 2, 1),
            ]
        )

        self.down_blocks = nn.ModuleList(
            [
                Block(down_channels[0], down_channels[0], self.cond_emb_dim),
                Block(down_channels[1], down_channels[1], self.cond_emb_dim),
                Block(down_channels[2], down_channels[2], self.cond_emb_dim),
            ]
        )

        self.bottleneck = Block(down_channels[-1], down_channels[-1], self.cond_emb_dim)

        self.upsample = nn.ModuleList(
            [
                nn.ConvTranspose1d(down_channels[2], down_channels[1], 4, 2, 1),
                nn.ConvTranspose1d(down_channels[1], down_channels[0], 4, 2, 1),
            ]
        )

        self.up_blocks = nn.ModuleList(
            [
                Block(down_channels[1] * 2, down_channels[1], self.cond_emb_dim),
                Block(down_channels[0] * 2, down_channels[0], self.cond_emb_dim),
            ]
        )

        # final Projection
        self.conv1 = nn.Conv1d(down_channels[0], mel_bins, 3, padding=1)

    def forward(self, x, time, content_features, speaker_embeddings):
        t_emb = self.time_mlp(time)
        cond_emb = torch.cat([t_emb, speaker_embeddings], dim=-1)

        # Content Conditioning
        # Ensure time dimensions align
        if content_features.shape[-1] != x.shape[-1]:
            print('INTERPOLATING')
            content_features = F.interpolate(
                content_features, size=x.shape[-1], mode='linear', align_corners=False
            )

        x = torch.cat([x, content_features], dim=1)

        x = self.conv0(x)
        r1 = x.clone()

        x = self.down_blocks[0](x, cond_emb)
        x = self.downsample[0](x)
        r2 = x.clone()

        x = self.down_blocks[1](x, cond_emb)
        x = self.downsample[1](x)

        x = self.down_blocks[2](x, cond_emb)
        x = self.bottleneck(x, cond_emb)

        x = self.upsample[0](x)
        x = torch.cat((x, r2), dim=1)
        x = self.up_blocks[0](x, cond_emb)

        x = self.upsample[1](x)
        x = torch.cat((x, r1), dim=1)
        x = self.up_blocks[1](x, cond_emb)

        x = self.conv1(x)
        return x
