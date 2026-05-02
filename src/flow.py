import torch
import torch.nn as nn
import torch.nn.functional as T
from tqdm.auto import tqdm, trange

from .data import inverse_normalization


def compute_loss(model, x1, device='cpu'):
    b, c, h, w = x1.shape

    x0 = torch.rand_like(x1).to(device)
    t = torch.rand(b).to(device)
    t_exp = t[..., None, None, None]  # or t.view(b, 1, 1, 1)

    xt = (1 - t_exp) * x0 + t_exp * x1

    # xt = (1 - t) x0 + t x1

    ut = x1 - x0  # actual
    vt = model(xt, t)  # pred

    loss = T.mse_loss(vt, ut)

    return loss


@torch.no_grad()
def sample_ode(model, shape, steps=500, device='cpu', leave_progress=False, store_all=False):
    x = torch.rand(shape).to(device)
    dt = 1 / steps
    if store_all:
        all_images = []

    for i in trange(steps, leave=leave_progress):
        t_val = i / steps
        t = torch.tensor([t_val]).to(device)  # torch.full([shape[0]], t_val)

        vt = model(x, t)

        x += vt * dt

        if store_all:
            all_images.append(x.cpu().clone().detach())

    if store_all:
        a = torch.stack(all_images)
        return a
    return x


def _generate(model, shape, device, mean, std, leave_progress=False, store_all=False):
    generated_images = sample_ode(
        model, shape, device=device, leave_progress=leave_progress, store_all=store_all
    )
    generated_images = inverse_normalization(generated_images.cpu(), mean, std)

    return generated_images
