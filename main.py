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
        '--dataset-length',
        default=75000,
        type=int,
        help='VCTK dataset length',
    )
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
    generate_group.add_argument('--input-audio-path', type=str, help='Path for the input audio')
    generate_group.add_argument(
        '--speaker-emb-path', type=str, help='Path for the speaker embedding file'
    )
    generate_group.add_argument(
        '--model-path',
        default='./model.pth',
        type=str,
        help='Path for the Model used for generation',
    )
    generate_group.add_argument(
        '--overwrite', action='store_true', help='Overwrite existing outputs without prompt'
    )

    args = parser.parse_args()
    validate_args(args)

    if args.train:
        train(
            length=args.dataset_length,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            lr=args.lr,
            epochs=args.epoch,
            device=args.device,
            checkpoint_path=args.checkpoint_path,
            save_path=args.save_path,
            precision=args.precision,
            resume_path=args.resume_checkpoint,
        )
    if args.generate:
        generate(
            input_audio_path=args.input_audio_path,
            speaker_emb_path=args.speaker_emb_path,
            model_path=args.model_path,
            device=args.device,
            save_path=args.save_path,
        )
