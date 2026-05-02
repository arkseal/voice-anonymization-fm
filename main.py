import argparse

from generate import generate
from src.utils import validate_args
from train import train

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='Main',
        description='This is the main file for the flow matching implementation',
    )
    parser.add_argument(
        '--device', default='cpu', type=str, help='Device used for training/generation'
    )
    parser.add_argument('--save-path', type=str, help='Path to save results')
    parser.add_argument(
        '--dataset-name',
        default='MNIST',
        type=str,
        help='Pytorch dataset being used for training/generation',
    )
    parser.add_argument(
        '--precision',
        default='full',
        help='Precision for model, use either full,half,amp,amp_bf16',
    )

    type_group = parser.add_mutually_exclusive_group()
    type_group.add_argument('--train', action='store_true', help='Train a model')
    type_group.add_argument(
        '--generate', action='store_true', help='Generate outputs based on the model'
    )

    train_group = parser.add_argument_group(title='Training', description='Values for training')
    train_group.add_argument(
        '--batch-size', default=1, type=int, help='Batch Size used for training'
    )
    train_group.add_argument(
        '--num-workers',
        default=0,
        type=int,
        help='Number of workers used for data loader',
    )
    train_group.add_argument(
        '--lr', default=1e-4, type=float, help='Learning rate used for training'
    )
    train_group.add_argument('--epoch', default=1, type=int, help='Number of epochs for training')
    train_group.add_argument(
        '--checkpoint-path',
        default='./checkpoints',
        type=str,
        help='Path to save checkpoints',
    )
    train_group.add_argument(
        '--resume-checkpoint', type=str, help='Path to checkpoint save to respond from'
    )

    generate_group = parser.add_argument_group(
        title='Generation', description='Values for generation'
    )
    generate_group.add_argument(
        '--model-path',
        default='./model.pth',
        type=str,
        help='Path for the Model used for generation',
    )
    generate_group.add_argument(
        '--num-channels', default=1, type=int, help='Number of channels for image'
    )
    generate_group.add_argument('--image-res', default=28, type=int, help='Image height/width')
    generate_group.add_argument(
        '--num-images', default=16, type=int, help='Number of images to generate'
    )
    generate_group.add_argument(
        '--images-per-row', default=4, type=int, help='Number of images per row'
    )

    args = parser.parse_args()
    validate_args(args)

    if args.train:
        train(
            args.dataset_name,
            args.batch_size,
            args.num_workers,
            args.lr,
            args.epoch,
            args.device,
            args.checkpoint_path,
            args.save_path,
            args.precision,
            args.resume_checkpoint,
        )
    if args.generate:
        shape = (args.num_images, args.num_channels, args.image_res, args.image_res)
        generate(
            args.model_path,
            shape,
            args.images_per_row,
            args.device,
            args.dataset_name,
            args.save_path,
        )
