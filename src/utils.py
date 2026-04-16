from pathlib import Path
from .data import DATASETS

import torch

_PRECISION = {
    'full': torch.float32,
    'half': torch.float16,
    'amp': torch.float16,
    'amp_bfloat16': torch.bfloat16,
    'amp_bf16': torch.bfloat16
}

def get_precision_dtype(precision):
    return _PRECISION.get(precision, torch.float32)

def _validate_training_args(args):
    if not args.batch_size >= 1:
        raise ValueError(f'Batch size must be at least 1, not {args.batch_size}')
    if not args.num_workers >= 0:
        raise ValueError(f'Num workers must be at least 0, not {args.num_workers}')
    if not args.epoch >= 1:
        raise ValueError(f'Epoch must be at least 1, not {args.epoch}')
    
    if not args.save_path:
        args.save_path = './results'
    args.save_path = Path(args.save_path)
    if not args.save_path.exists():
        print(f"Save path {args.save_path} doesn't exist, making folder")
        args.save_path.mkdir()
    
    if args.resume_checkpoint and not Path(args.resume_checkpoint).exists():
        raise ValueError(f"Checkpoint resume path {args.resume_checkpoint} doesn't exist")

    args.checkpoint_path = Path(args.checkpoint_path)
    if not args.checkpoint_path.exists():
        print(f"Checkpoint path {args.checkpoint_path} doesn't exist, making folder")
        args.checkpoint_path.mkdir()

def _validate_generation_args(args):
    if args.num_channels not in (1, 3):
        raise ValueError(f'Number of channels must be either 1 or 3, not {args.num_channels}')
    if not Path(args.model_path).exists():
        raise ValueError(f'Model path must exist, {args.model_path} not found')
    
    if not args.save_path:
        args.save_path = './results.png'
    args.save_path = Path(args.save_path)
    if args.save_path.exists():
        if 'y' not in input(f'Save path {args.save_path} already exists, overwrite? (Y/N): ').lower():
            raise ValueError(f'Save path {args.save_path} already exists')

def validate_args(args):
    if not args.train and not args.generate:
        raise ValueError('Either train flag or generate flag must be used')
    if args.device == 'cuda':
        if not torch.cuda.is_available():
            raise ValueError('CUDA device was selected but CUDA is not availible')
    elif args.device == 'xpu':
        if not torch.xpu.is_available():
            raise ValueError('XPU device was selected but XPU is not availible')
    
    if args.dataset_name not in DATASETS:
        raise ValueError(f'Dataset name must be in {DATASETS.keys()}, not {args.dataset_name}')
    if args.precision not in _PRECISION:
        print(f'{args.precision} precision unknown, using full precision')

    if args.train:
        _validate_training_args(args)
    elif args.generate:
        _validate_generation_args(args)
