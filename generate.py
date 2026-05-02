from pathlib import Path

import torch
import torchvision.transforms.v2 as T
from torchvision.utils import make_grid, save_image

from src.data import get_norm
from src.flow import _generate
from src.model import FlowMatchingUNet


def generate(
    model_path,
    shape,
    nrow,
    device,
    dataset_name,
    save_path=Path('./results.png'),
    generate_gif=True,
):
    print('Loading Model...')
    norm = get_norm(dataset_name)
    model = FlowMatchingUNet(shape[1])

    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict['model_state_dict'])

    model.to(device)
    model.eval()
    print('Loaded Model')

    print('Generating Samples...')
    generated_images = _generate(
        model, shape, device, **norm, leave_progress=True, store_all=generate_gif
    )

    save_image(
        make_grid(generated_images[-1] if generate_gif else generated_images, nrow=nrow),
        save_path,
    )
    if generate_gif:
        grids = torch.stack([make_grid(batch, nrow=nrow) for batch in generated_images])
        grids = grids.mul(255).add_(0.5).clamp_(0, 255).to(torch.uint8)

        transforms = T.Compose([T.RGB(), T.ToPILImage()])

        pil_images = [transforms(img) for img in grids]

        pil_images[0].save(
            save_path.parent / (save_path.stem + '.gif'),
            format='GIF',
            save_all=True,
            optimize=False,
            append_images=[
                im for i, im in enumerate(pil_images[1:], 1) if i % 6 == 0 or i == len(pil_images)
            ],
            duration=20,
            loop=0,  # 0 means the GIF will loop infinitely
        )
