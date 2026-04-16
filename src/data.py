import torch
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T
from torchvision.datasets import MNIST, CIFAR10

DATASETS = {
    'MNIST': {
        'normalization': {
            # Normalize [0, 1] to [-1, 1]
            'mean': [0.5],
            'std': [0.5],
        },
        'randomization': [],
        'dataset': MNIST,
        'shape': (1, 28, 28)
    },
    'CIFAR10': {
        'normalization': {
            'mean': [0.4914, 0.4822, 0.4465],
            'std': [0.2470, 0.2435, 0.2616],
        },
        'randomization': [
            T.RandomCrop(32, 4, padding_mode='reflect'),
            T.RandomHorizontalFlip()
        ],
        'dataset': CIFAR10,
        'shape': (3, 32, 32)
    }
}

def calculate_normalization_stats(dataset, batch_size=1024):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    sample = dataset[0][0]
    
    mean = torch.zeros([sample.size(0)])
    std = torch.zeros([sample.size(0)])
    total = 0
    for batch, _ in dataloader:
        batch = batch.view(batch.size(0), batch.size(1), -1)
        
        mean += batch.mean(dim=(0, 2))
        std += batch.std(dim=(0, 2))
        total += batch.size(0)
    
    mean /= total
    std /= total
    
    return mean, std

def inverse_normalization(x, mean, std):
    if not isinstance(mean, list):
        mean = [mean]
        std = [std]
    mean_t = torch.tensor(mean, dtype=x.dtype, device=x.device)
    std_t = torch.tensor(std, dtype=x.dtype, device=x.device)
    
    channel_dim = x.dim() - 3
    shape = [1] * x.dim()
    shape[channel_dim] = -1
    
    x = x * std_t.view(*shape) + mean_t.view(*shape)

    return x

def get_norm(dataset_name):
    return DATASETS[dataset_name]['normalization']

def get_train_data(dataset_name, batch_size=128, num_workers=0):
    transform = T.Compose([
        *DATASETS[dataset_name]['randomization'],
        T.ToTensor(),
        T.Normalize(**DATASETS[dataset_name]['normalization'])
    ])
    
    dataset = DATASETS[dataset_name]['dataset'](root='./data', train=True, transform=transform, download=True)
    
    dataloader = DataLoader(dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers, pin_memory=True, persistent_workers=True)
    
    return dataloader, DATASETS[dataset_name]['shape'], DATASETS[dataset_name]['normalization']
